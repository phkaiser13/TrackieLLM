/*
 * Copyright (C) 2025 Pedro Henrique / phdev13
 *
 * File: tk_contextual_reasoner.c
 *
 * This file contains the implementation of the Contextual Reasoning Engine.
 * The engine aggregates multimodal information, scores relevance, decays old
 * context, and produces a compact representation for the LLM.
 *
 * SPDX-License-Identifier: AGPL-3.0 license Apache-2.0
 */

#include "cortex/tk_contextual_reasoner.h"
#include "utils/tk_logging.h"
#include "utils/tk_error_handling.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <pthread.h>
#include <time.h>
#include <unistd.h>   /* for sysconf(_SC_PAGESIZE) – optional */

/* -------------------------------------------------------------------------- */
/* Internal data structures                                                   */
/* -------------------------------------------------------------------------- */

/**
 * @struct context_memory_t
 * @brief Stores generic context items (environment, navigation, user, …).
 */
typedef struct context_memory_s {
    tk_context_item_t *items;          /**< Array of items (circular buffer). */
    size_t             item_count;    /**< Number of valid items. */
    size_t             capacity;      /**< Maximum number of items. */
    size_t             next_index;    /**< Index where the next item will be placed. */
    pthread_mutex_t    mutex;         /**< Protects the whole structure. */
} context_memory_t;

/**
 * @struct conversation_memory_t
 * @brief Stores the dialogue history.
 */
typedef struct conversation_memory_s {
    tk_conversation_turn_t *turns;     /**< Circular buffer of turns. */
    size_t                  turn_count;
    size_t                  capacity;
    size_t                  next_index;
    pthread_mutex_t         mutex;
} conversation_memory_t;

/**
 * @struct tk_contextual_reasoner_s
 * @brief Main reasoner object.
 */
struct tk_contextual_reasoner_s {
    tk_context_config_t config;                /**< User‑provided configuration. */

    /* ------------------------------------------------------------------ */
    /*  Memory containers                                                  */
    /* ------------------------------------------------------------------ */
    context_memory_t      context_memory;
    conversation_memory_t conversation_memory;

    /* ------------------------------------------------------------------ */
    /*  Environmental state (vision)                                       */
    /* ------------------------------------------------------------------ */
    struct {
        tk_vision_object_t *visible_objects;
        size_t              visible_object_count;
        size_t              visible_object_capacity;
        uint64_t            last_update_ns;
    } environmental_state;

    /* ------------------------------------------------------------------ */
    /*  Audio state                                                        */
    /* ------------------------------------------------------------------ */
    struct {
        tk_ambient_sound_type_e last_detected_sound;
        uint64_t                last_sound_update_ns;
    } audio_state;

    /* ------------------------------------------------------------------ */
    /*  Navigation state                                                   */
    /* ------------------------------------------------------------------ */
    struct {
        bool                has_clear_path;
        float               clear_path_direction_deg;
        float               clear_path_distance_m;
        tk_navigation_hazard_t *hazards;
        size_t              hazard_count;
        size_t              hazard_capacity;
        tk_motion_state_e   motion_state; /* From sensor fusion */
        tk_navigation_cue_type_e last_detected_cue;
        uint64_t            last_update_ns;
    } navigation_state;

    /* ------------------------------------------------------------------ */
    /*  System‑wide state                                                  */
    /* ------------------------------------------------------------------ */
    struct {
        uint64_t last_context_process_ns;
        uint64_t total_memory_bytes;          /**< Approx. bytes used by context. */
        bool     is_listening_for_commands;
        float    system_confidence;
    } system_state;

    /* ------------------------------------------------------------------ */
    /*  Global lock – protects the three state structs above               */
    /* ------------------------------------------------------------------ */
    pthread_mutex_t state_mutex;

    /* ------------------------------------------------------------------ */
    /*  Optional critical‑event callback (not part of the public header)   */
    /* ------------------------------------------------------------------ */
    tk_critical_event_cb_t critical_cb;
    void *critical_user_data;
};

/* -------------------------------------------------------------------------- */
/* Forward declarations of internal helpers                                   */
/* -------------------------------------------------------------------------- */
static tk_error_code_t init_context_memory(context_memory_t *mem, size_t capacity);
static void           cleanup_context_memory(context_memory_t *mem);
static tk_error_code_t init_conversation_memory(conversation_memory_t *mem,
                                                size_t capacity);
static void           cleanup_conversation_memory(conversation_memory_t *mem);

static tk_error_code_t add_context_item_internal(tk_contextual_reasoner_t *r,
                                                const tk_context_item_t *src);
static void           update_context_relevance_scores(tk_contextual_reasoner_t *r,
                                                      uint64_t now_ns);
static void           prune_irrelevant_context(tk_contextual_reasoner_t *r);
static float          calculate_relevance_score(const tk_context_item_t *item,
                                                uint64_t now_ns,
                                                float decay_rate);

static char *generate_environmental_description(tk_contextual_reasoner_t *r);
static char *generate_navigation_description(tk_contextual_reasoner_t *r);
static char *generate_conversation_summary(tk_contextual_reasoner_t *r,
                                           size_t max_turns);

static uint64_t get_current_time_ns(void);

/* -------------------------------------------------------------------------- */
/* Public API implementation                                                  */
/* -------------------------------------------------------------------------- */

tk_error_code_t tk_contextual_reasoner_create(tk_contextual_reasoner_t **out,
                                              const tk_context_config_t *cfg)
{
    if (!out || !cfg) return TK_ERROR_INVALID_ARGUMENT;

    tk_log_info("Creating Contextual Reasoning Engine");

    tk_contextual_reasoner_t *r = calloc(1, sizeof(*r));
    if (!r) return TK_ERROR_OUT_OF_MEMORY;

    r->config = *cfg;

    /* initialise containers ------------------------------------------------ */
    tk_error_code_t rc = init_context_memory(&r->context_memory,
                                             cfg->max_context_history_items);
    if (rc != TK_SUCCESS) { free(r); return rc; }

    rc = init_conversation_memory(&r->conversation_memory,
                                  cfg->max_conversation_history_turns);
    if (rc != TK_SUCCESS) {
        cleanup_context_memory(&r->context_memory);
        free(r);
        return rc;
    }

    /* environmental state --------------------------------------------------- */
    r->audio_state.last_detected_sound = TK_AMBIENT_SOUND_NONE;
    r->environmental_state.visible_object_capacity = 64;
    r->environmental_state.visible_objects = calloc(
        r->environmental_state.visible_object_capacity,
        sizeof(tk_vision_object_t));
    if (!r->environmental_state.visible_objects) {
        cleanup_conversation_memory(&r->conversation_memory);
        cleanup_context_memory(&r->context_memory);
        free(r);
        return TK_ERROR_OUT_OF_MEMORY;
    }

    /* navigation state ------------------------------------------------------ */
    r->navigation_state.motion_state = TK_MOTION_STATE_UNKNOWN;
    r->navigation_state.last_detected_cue = TK_NAVIGATION_CUE_NONE;
    r->navigation_state.hazard_capacity = 32;
    r->navigation_state.hazards = calloc(r->navigation_state.hazard_capacity,
                                         sizeof(tk_navigation_hazard_t));
    if (!r->navigation_state.hazards) {
        free(r->environmental_state.visible_objects);
        cleanup_conversation_memory(&r->conversation_memory);
        cleanup_context_memory(&r->context_memory);
        free(r);
        return TK_ERROR_OUT_OF_MEMORY;
    }

    /* global mutex ---------------------------------------------------------- */
    if (pthread_mutex_init(&r->state_mutex, NULL) != 0) {
        free(r->navigation_state.hazards);
        free(r->environmental_state.visible_objects);
        cleanup_conversation_memory(&r->conversation_memory);
        cleanup_context_memory(&r->context_memory);
        free(r);
        return TK_ERROR_SYSTEM_ERROR;
    }

    /* default system state -------------------------------------------------- */
    r->system_state.system_confidence = 0.8f;
    r->system_state.is_listening_for_commands = false;
    r->system_state.total_memory_bytes = 0;
    r->critical_cb = NULL;
    r->critical_user_data = NULL;

    *out = r;
    tk_log_info("Contextual Reasoning Engine created");
    return TK_SUCCESS;
}

/* ---------------------------------------------------------------------- */

tk_error_code_t tk_contextual_reasoner_get_motion_state(
    tk_contextual_reasoner_t* reasoner,
    tk_motion_state_e* out_state)
{
    if (!reasoner || !out_state) {
        return TK_ERROR_INVALID_ARGUMENT;
    }

    pthread_mutex_lock(&reasoner->state_mutex);
    *out_state = reasoner->navigation_state.motion_state;
    pthread_mutex_unlock(&reasoner->state_mutex);

    return TK_SUCCESS;
}

/* ---------------------------------------------------------------------- */

tk_error_code_t tk_contextual_reasoner_update_ambient_sound(
    tk_contextual_reasoner_t* reasoner,
    tk_ambient_sound_type_e sound_type,
    float confidence)
{
    if (!reasoner) {
        return TK_ERROR_INVALID_ARGUMENT;
    }

    pthread_mutex_lock(&reasoner->state_mutex);
    reasoner->audio_state.last_detected_sound = sound_type;
    reasoner->audio_state.last_sound_update_ns = get_current_time_ns();
    pthread_mutex_unlock(&reasoner->state_mutex);

    if (sound_type != TK_AMBIENT_SOUND_NONE) {
        char description[128];
        const char* sound_str = "Unknown sound";
        tk_context_priority_e priority = TK_CONTEXT_PRIORITY_MEDIUM;

        switch(sound_type) {
            case TK_AMBIENT_SOUND_FIRE_ALARM:
                sound_str = "Fire alarm detected";
                priority = TK_CONTEXT_PRIORITY_CRITICAL;
                break;
            case TK_AMBIENT_SOUND_SIREN:
                sound_str = "Siren detected";
                priority = TK_CONTEXT_PRIORITY_HIGH;
                break;
            case TK_AMBIENT_SOUND_CAR_HORN:
                sound_str = "Car horn detected";
                priority = TK_CONTEXT_PRIORITY_HIGH;
                break;
            case TK_AMBIENT_SOUND_BABY_CRYING:
                sound_str = "Baby crying detected";
                priority = TK_CONTEXT_PRIORITY_MEDIUM;
                break;
            case TK_AMBIENT_SOUND_DOORBELL:
                sound_str = "Doorbell detected";
                priority = TK_CONTEXT_PRIORITY_LOW;
                break;
            default:
                break;
        }

        snprintf(description, sizeof(description), "%s (confidence: %.0f%%)", sound_str, confidence * 100.0f);

        tk_contextual_reasoner_add_context_item(reasoner,
            TK_CONTEXT_TYPE_ENVIRONMENTAL,
            priority,
            description,
            NULL, 0);
    }

    return TK_SUCCESS;
}

/* ---------------------------------------------------------------------- */

tk_error_code_t tk_contextual_reasoner_update_navigation_cues(
    tk_contextual_reasoner_t* reasoner,
    tk_navigation_cue_type_e cue_type,
    float distance_m)
{
    if (!reasoner) {
        return TK_ERROR_INVALID_ARGUMENT;
    }

    pthread_mutex_lock(&reasoner->state_mutex);
    reasoner->navigation_state.last_detected_cue = cue_type;
    pthread_mutex_unlock(&reasoner->state_mutex);

    if (cue_type != TK_NAVIGATION_CUE_NONE) {
        char description[128];
        const char* cue_str = "Unknown navigation cue";
        tk_context_priority_e priority = TK_CONTEXT_PRIORITY_HIGH;

        switch(cue_type) {
            case TK_NAVIGATION_CUE_STEP_UP:
                cue_str = "Step up detected";
                break;
            case TK_NAVIGATION_CUE_STEP_DOWN:
                cue_str = "Step down detected";
                break;
            case TK_NAVIGATION_CUE_DOORWAY:
                cue_str = "Doorway detected";
                priority = TK_CONTEXT_PRIORITY_MEDIUM;
                break;
            case TK_NAVIGATION_CUE_STAIRS_UP:
                cue_str = "Stairs up detected";
                break;
            case TK_NAVIGATION_CUE_STAIRS_DOWN:
                cue_str = "Stairs down detected";
                break;
            default:
                break;
        }

        snprintf(description, sizeof(description), "%s at %.1fm", cue_str, distance_m);

        tk_contextual_reasoner_add_context_item(reasoner,
            TK_CONTEXT_TYPE_NAVIGATIONAL,
            priority,
            description,
            NULL, 0);
    }

    return TK_SUCCESS;
}

/* ---------------------------------------------------------------------- */

void tk_contextual_reasoner_destroy(tk_contextual_reasoner_t **reasoner)
{
    if (!reasoner || !*reasoner) return;

    tk_log_info("Destroying Contextual Reasoning Engine");
    tk_contextual_reasoner_t *r = *reasoner;

    /* free containers ----------------------------------------------------- */
    cleanup_conversation_memory(&r->conversation_memory);
    cleanup_context_memory(&r->context_memory);

    /* free state arrays ---------------------------------------------------- */
    free(r->environmental_state.visible_objects);
    free(r->navigation_state.hazards);

    /* destroy mutexes ------------------------------------------------------ */
    pthread_mutex_destroy(&r->state_mutex);

    free(r);
    *reasoner = NULL;
    tk_log_info("Contextual Reasoning Engine destroyed");
}

/* ---------------------------------------------------------------------- */

tk_error_code_t tk_contextual_reasoner_update_vision_context(
        tk_contextual_reasoner_t *reasoner,
        const tk_vision_result_t *vision_result)
{
    if (!reasoner || !vision_result) return TK_ERROR_INVALID_ARGUMENT;

    /* ------------------------------------------------------------------ */
    /*  Update the fast‑path environmental snapshot (visible objects)      */
    /* ------------------------------------------------------------------ */
    pthread_mutex_lock(&reasoner->state_mutex);

    size_t copy_cnt = vision_result->object_count;
    if (copy_cnt > reasoner->environmental_state.visible_object_capacity)
        copy_cnt = reasoner->environmental_state.visible_object_capacity;

    if (copy_cnt)
        memcpy(reasoner->environmental_state.visible_objects,
               vision_result->objects,
               copy_cnt * sizeof(tk_vision_object_t));

    reasoner->environmental_state.visible_object_count = copy_cnt;
    reasoner->environmental_state.last_update_ns = get_current_time_ns();

    pthread_mutex_unlock(&reasoner->state_mutex);

    /* ------------------------------------------------------------------ */
    /*  Create high‑level context items for the reasoner                 */
    /* ------------------------------------------------------------------ */
    for (size_t i = 0; i < copy_cnt; ++i) {
        const tk_vision_object_t *obj = &vision_result->objects[i];

        if (obj->confidence < 0.7f) continue;   /* ignore low‑confidence */

        char desc[256];
        snprintf(desc, sizeof(desc),
                 "Detected %s at %.1fm (confidence %.0f%%)",
                 obj->label,
                 obj->distance_meters,
                 obj->confidence * 100.0f);

        tk_context_item_t item = {
            .timestamp_ns   = get_current_time_ns(),
            .type           = TK_CONTEXT_TYPE_ENVIRONMENTAL,
            .priority       = (obj->distance_meters < 2.0f)
                              ? TK_CONTEXT_PRIORITY_HIGH
                              : TK_CONTEXT_PRIORITY_MEDIUM,
            .relevance_score = obj->confidence,
            .description    = NULL,
            .data           = NULL,
            .data_size      = 0
        };
        item.description = strdup(desc);
        if (!item.description) return TK_ERROR_OUT_OF_MEMORY;

        add_context_item_internal(reasoner, &item);
        /* internal function makes its own copy, we can free the temporary */
        free(item.description);
    }

    return TK_SUCCESS;
}

/* ---------------------------------------------------------------------- */

tk_error_code_t tk_contextual_reasoner_update_navigation_context(
        tk_contextual_reasoner_t *reasoner,
        const tk_traversability_map_t *traversability_map,
        const tk_free_space_analysis_t *free_space_analysis,
        const tk_obstacle_t *obstacles,
        size_t obstacle_count)
{
    if (!reasoner || !traversability_map || !free_space_analysis)
        return TK_ERROR_INVALID_ARGUMENT;

    pthread_mutex_lock(&reasoner->state_mutex);

    /* update navigation snapshot */
    reasoner->navigation_state.has_clear_path = free_space_analysis->is_any_path_clear;
    reasoner->navigation_state.clear_path_direction_deg =
        free_space_analysis->clearest_path_angle_deg;
    reasoner->navigation_state.clear_path_distance_m =
        free_space_analysis->clearest_path_distance_m;
    reasoner->navigation_state.last_update_ns = get_current_time_ns();

    /* reset hazard list – will be filled later if needed */
    reasoner->navigation_state.hazard_count = 0;

    pthread_mutex_unlock(&reasoner->state_mutex);

    /* ------------------------------------------------------------------ */
    /*  High‑level navigation context items                              */
    /* ------------------------------------------------------------------ */
    if (free_space_analysis->is_any_path_clear) {
        char buf[256];
        snprintf(buf, sizeof(buf),
                 "Clear path at %.0f°, distance %.1fm",
                 free_space_analysis->clearest_path_angle_deg,
                 free_space_analysis->clearest_path_distance_m);
        tk_contextual_reasoner_add_context_item(reasoner,
                                                TK_CONTEXT_TYPE_NAVIGATIONAL,
                                                TK_CONTEXT_PRIORITY_HIGH,
                                                buf,
                                                NULL, 0);
    } else {
        tk_contextual_reasoner_add_context_item(reasoner,
                                                TK_CONTEXT_TYPE_NAVIGATIONAL,
                                                TK_CONTEXT_PRIORITY_CRITICAL,
                                                "No clear navigation path detected",
                                                NULL, 0);
    }

    /* ------------------------------------------------------------------ */
    /*  Obstacles – we only keep the first few most relevant ones       */
    /* ------------------------------------------------------------------ */
    size_t max_obstacles = (obstacle_count < 5) ? obstacle_count : 5;
    for (size_t i = 0; i < max_obstacles; ++i) {
        const tk_obstacle_t *obs = &obstacles[i];
        char buf[256];
        snprintf(buf, sizeof(buf),
                 "Obstacle at (%.1f, %.1f)m size %.1fx%.1fm",
                 obs->position_m.x, obs->position_m.y,
                 obs->dimensions_m.x, obs->dimensions_m.y);

        /* priority rises when the obstacle is close */
        float dist = sqrtf(obs->position_m.x * obs->position_m.x +
                          obs->position_m.y * obs->position_m.y);
        tk_context_priority_e prio = (dist < 1.5f)
                                    ? TK_CONTEXT_PRIORITY_HIGH
                                    : TK_CONTEXT_PRIORITY_MEDIUM;

        tk_contextual_reasoner_add_context_item(reasoner,
                                                TK_CONTEXT_TYPE_NAVIGATIONAL,
                                                prio,
                                                buf,
                                                NULL, 0);
    }

    return TK_SUCCESS;
}

/* ---------------------------------------------------------------------- */

tk_error_code_t tk_contextual_reasoner_add_conversation_turn(
        tk_contextual_reasoner_t *reasoner,
        bool is_user_input,
        const char *content,
        float confidence)
{
    if (!reasoner || !content) return TK_ERROR_INVALID_ARGUMENT;

    conversation_memory_t *mem = &reasoner->conversation_memory;
    pthread_mutex_lock(&mem->mutex);

    size_t idx = mem->next_index;
    tk_conversation_turn_t *slot = &mem->turns[idx];

    /* free previous content if we are overwriting an old entry */
    if (slot->content) {
        free(slot->content);
        slot->content = NULL;
    }

    slot->timestamp_ns   = get_current_time_ns();
    slot->is_user_input  = is_user_input;
    slot->content        = strdup(content);
    slot->confidence     = confidence;

    if (!slot->content) {
        pthread_mutex_unlock(&mem->mutex);
        return TK_ERROR_OUT_OF_MEMORY;
    }

    /* advance circular buffer */
    mem->next_index = (mem->next_index + 1) % mem->capacity;
    if (mem->turn_count < mem->capacity) mem->turn_count++;

    pthread_mutex_unlock(&mem->mutex);
    return TK_SUCCESS;
}

/* ---------------------------------------------------------------------- */

tk_error_code_t tk_contextual_reasoner_add_context_item(
        tk_contextual_reasoner_t *reasoner,
        tk_context_type_e type,
        tk_context_priority_e priority,
        const char *description,
        const void *data,
        size_t data_size)
{
    if (!reasoner || !description) return TK_ERROR_INVALID_ARGUMENT;

    tk_context_item_t src = {
        .timestamp_ns    = get_current_time_ns(),
        .type            = type,
        .priority        = priority,
        .relevance_score = 1.0f,          /* fresh items start fully relevant */
        .description     = NULL,
        .data            = NULL,
        .data_size       = 0
    };

    src.description = strdup(description);
    if (!src.description) return TK_ERROR_OUT_OF_MEMORY;

    if (data && data_size) {
        src.data = malloc(data_size);
        if (!src.data) {
            free(src.description);
            return TK_ERROR_OUT_OF_MEMORY;
        }
        memcpy((void *)src.data, data, data_size);
        src.data_size = data_size;
    }

    tk_error_code_t rc = add_context_item_internal(reasoner, &src);

    /* internal function makes its own deep copy – we can free temporaries */
    free(src.description);
    free((void *)src.data);
    return rc;
}

/* ---------------------------------------------------------------------- */

tk_error_code_t tk_contextual_reasoner_process_context(
        tk_contextual_reasoner_t *reasoner,
        uint64_t current_time_ns)
{
    if (!reasoner) return TK_ERROR_INVALID_ARGUMENT;

    /* 1) update relevance scores (decay) */
    update_context_relevance_scores(reasoner, current_time_ns);

    /* 2) prune items that fell below the relevance threshold */
    prune_irrelevant_context(reasoner);

    /* 3) update bookkeeping */
    pthread_mutex_lock(&reasoner->state_mutex);
    reasoner->system_state.last_context_process_ns = current_time_ns;
    pthread_mutex_unlock(&reasoner->state_mutex);

    return TK_SUCCESS;
}

/* ---------------------------------------------------------------------- */

tk_error_code_t tk_contextual_reasoner_get_context_summary(
        tk_contextual_reasoner_t *reasoner,
        tk_context_summary_t *out_summary)
{
    if (!reasoner || !out_summary) return TK_ERROR_INVALID_ARGUMENT;

    /* Zero‑initialise the output – callers may rely on NULL pointers */
    memset(out_summary, 0, sizeof(*out_summary));

    /* ------------------------------------------------------------------ */
    /*  Environmental part                                                */
    /* ------------------------------------------------------------------ */
    pthread_mutex_lock(&reasoner->state_mutex);
    out_summary->visible_object_count = reasoner->environmental_state.visible_object_count;
    out_summary->visible_objects = reasoner->environmental_state.visible_objects;
    pthread_mutex_unlock(&reasoner->state_mutex);

    /* ------------------------------------------------------------------ */
    /*  Navigation part                                                   */
    /* ------------------------------------------------------------------ */
    pthread_mutex_lock(&reasoner->state_mutex);
    out_summary->has_clear_path = reasoner->navigation_state.has_clear_path;
    out_summary->clear_path_direction_deg = reasoner->navigation_state.clear_path_direction_deg;
    out_summary->clear_path_distance_m = reasoner->navigation_state.clear_path_distance_m;
    out_summary->hazard_count = reasoner->navigation_state.hazard_count;
    out_summary->hazards = reasoner->navigation_state.hazards;
    pthread_mutex_unlock(&reasoner->state_mutex);

    /* ------------------------------------------------------------------ */
    /*  Conversation part                                                 */
    /* ------------------------------------------------------------------ */
    conversation_memory_t *c_mem = &reasoner->conversation_memory;
    pthread_mutex_lock(&c_mem->mutex);
    out_summary->conversation_turn_count = c_mem->turn_count;
    out_summary->recent_conversation = c_mem->turns;   /* circular buffer – read‑only */
    pthread_mutex_unlock(&c_mem->mutex);

    /* ------------------------------------------------------------------ */
    /*  Temporal / system part                                            */
    /* ------------------------------------------------------------------ */
    pthread_mutex_lock(&reasoner->state_mutex);
    out_summary->recent_events_summary = NULL; /* optional – left for future use */
    out_summary->is_navigation_active = reasoner->navigation_state.has_clear_path;
    out_summary->is_listening_for_commands = reasoner->system_state.is_listening_for_commands;
    out_summary->system_confidence = reasoner->system_state.system_confidence;
    out_summary->user_motion_state = reasoner->navigation_state.motion_state;
    out_summary->detected_sound_type = reasoner->audio_state.last_detected_sound;
    out_summary->detected_navigation_cue = reasoner->navigation_state.last_detected_cue;
    pthread_mutex_unlock(&reasoner->state_mutex);

    return TK_SUCCESS;
}

/* ---------------------------------------------------------------------- */

tk_error_code_t tk_contextual_reasoner_generate_context_string(
        tk_contextual_reasoner_t *reasoner,
        char **out_context_string,
        size_t max_token_budget)
{
    if (!reasoner || !out_context_string) return TK_ERROR_INVALID_ARGUMENT;
    *out_context_string = NULL;

    /* Approximate characters per token – 4 is a safe average for English/PT */
    size_t max_chars = max_token_budget * 4;
    size_t used = 0;
    char *buf = calloc(1, max_chars + 1);
    if (!buf) return TK_ERROR_OUT_OF_MEMORY;

    /* ------------------------------------------------------------------ */
    /*  1) Environmental description                                      */
    /* ------------------------------------------------------------------ */
    char *env = generate_environmental_description(reasoner);
    if (env) {
        size_t len = strlen(env);
        if (used + len + 1 <= max_chars) {
            memcpy(buf + used, env, len);
            used += len;
            buf[used++] = ' ';
        }
        free(env);
    }

    /* ------------------------------------------------------------------ */
    /*  2) Navigation description                                         */
    /* ------------------------------------------------------------------ */
    char *nav = generate_navigation_description(reasoner);
    if (nav) {
        size_t len = strlen(nav);
        if (used + len + 1 <= max_chars) {
            memcpy(buf + used, nav, len);
            used += len;
            buf[used++] = ' ';
        }
        free(nav);
    }

    /* ------------------------------------------------------------------ */
    /*  3) Recent conversation (up to a few turns)                       */
    /* ------------------------------------------------------------------ */
    char *conv = generate_conversation_summary(reasoner, 3);
    if (conv) {
        size_t len = strlen(conv);
        if (used + len + 1 <= max_chars) {
            memcpy(buf + used, conv, len);
            used += len;
            buf[used++] = ' ';
        }
        free(conv);
    }

    /* Trim trailing space */
    if (used && buf[used - 1] == ' ') buf[--used] = '\0';
    else buf[used] = '\0';

    *out_context_string = buf;
    return TK_SUCCESS;
}

/* ---------------------------------------------------------------------- */

tk_error_code_t tk_contextual_reasoner_free_context_string(char *ptr)
{
    if (!ptr) return TK_ERROR_INVALID_ARGUMENT;
    free(ptr);
    return TK_SUCCESS;
}

/* ---------------------------------------------------------------------- */

tk_error_code_t tk_contextual_reasoner_clear_context(
        tk_contextual_reasoner_t *reasoner)
{
    if (!reasoner) return TK_ERROR_INVALID_ARGUMENT;

    /* clear generic context items */
    pthread_mutex_lock(&reasoner->context_memory.mutex);
    for (size_t i = 0; i < reasoner->context_memory.item_count; ++i) {
        tk_context_item_t *it = &reasoner->context_memory.items[i];
        free(it->description);
        free(it->data);
        memset(it, 0, sizeof(*it));
    }
    reasoner->context_memory.item_count = 0;
    reasoner->context_memory.next_index = 0;
    pthread_mutex_unlock(&reasoner->context_memory.mutex);

    /* clear conversation history */
    pthread_mutex_lock(&reasoner->conversation_memory.mutex);
    for (size_t i = 0; i < reasoner->conversation_memory.turn_count; ++i) {
        tk_conversation_turn_t *t = &reasoner->conversation_memory.turns[i];
        free(t->content);
        t->content = NULL;
    }
    reasoner->conversation_memory.turn_count = 0;
    reasoner->conversation_memory.next_index = 0;
    pthread_mutex_unlock(&reasoner->conversation_memory.mutex);

    /* reset system bookkeeping */
    pthread_mutex_lock(&reasoner->state_mutex);
    reasoner->system_state.total_memory_bytes = 0;
    reasoner->system_state.last_context_process_ns = 0;
    pthread_mutex_unlock(&reasoner->state_mutex);

    return TK_SUCCESS;
}

/* ---------------------------------------------------------------------- */

tk_error_code_t tk_contextual_reasoner_get_memory_stats(
        tk_contextual_reasoner_t *reasoner,
        size_t *out_total_items,
        size_t *out_total_memory_bytes,
        size_t *out_conversation_turns)
{
    if (!reasoner) return TK_ERROR_INVALID_ARGUMENT;

    if (out_total_items) {
        pthread_mutex_lock(&reasoner->context_memory.mutex);
        *out_total_items = reasoner->context_memory.item_count;
        pthread_mutex_unlock(&reasoner->context_memory.mutex);
    }

    if (out_conversation_turns) {
        pthread_mutex_lock(&reasoner->conversation_memory.mutex);
        *out_conversation_turns = reasoner->conversation_memory.turn_count;
        pthread_mutex_unlock(&reasoner->conversation_memory.mutex);
    }

    if (out_total_memory_bytes) {
        size_t bytes = 0;
        pthread_mutex_lock(&reasoner->context_memory.mutex);
        for (size_t i = 0; i < reasoner->context_memory.item_count; ++i) {
            const tk_context_item_t *it = &reasoner->context_memory.items[i];
            bytes += sizeof(*it);
            if (it->description) bytes += strlen(it->description) + 1;
            if (it->data) bytes += it->data_size;
        }
        pthread_mutex_unlock(&reasoner->context_memory.mutex);
        *out_total_memory_bytes = bytes;
    }

    return TK_SUCCESS;
}

/* -------------------------------------------------------------------------- */
/* Internal helper implementations                                            */
/* -------------------------------------------------------------------------- */

/* ---------- context memory ------------------------------------------------- */
static tk_error_code_t init_context_memory(context_memory_t *mem,
                                           size_t capacity)
{
    if (!mem || capacity == 0) return TK_ERROR_INVALID_ARGUMENT;

    mem->items = calloc(capacity, sizeof(tk_context_item_t));
    if (!mem->items) return TK_ERROR_OUT_OF_MEMORY;

    mem->capacity   = capacity;
    mem->item_count = 0;
    mem->next_index = 0;
    if (pthread_mutex_init(&mem->mutex, NULL) != 0) {
        free(mem->items);
        return TK_ERROR_SYSTEM_ERROR;
    }
    return TK_SUCCESS;
}

static void cleanup_context_memory(context_memory_t *mem)
{
    if (!mem) return;
    pthread_mutex_lock(&mem->mutex);
    for (size_t i = 0; i < mem->item_count; ++i) {
        tk_context_item_t *it = &mem->items[i];
        free(it->description);
        free(it->data);
    }
    free(mem->items);
    mem->items = NULL;
    mem->capacity = mem->item_count = mem->next_index = 0;
    pthread_mutex_unlock(&mem->mutex);
    pthread_mutex_destroy(&mem->mutex);
}

/* ---------- conversation memory ------------------------------------------- */
static tk_error_code_t init_conversation_memory(conversation_memory_t *mem,
                                                size_t capacity)
{
    if (!mem || capacity == 0) return TK_ERROR_INVALID_ARGUMENT;

    mem->turns = calloc(capacity, sizeof(tk_conversation_turn_t));
    if (!mem->turns) return TK_ERROR_OUT_OF_MEMORY;

    mem->capacity   = capacity;
    mem->turn_count = 0;
    mem->next_index = 0;
    if (pthread_mutex_init(&mem->mutex, NULL) != 0) {
        free(mem->turns);
        return TK_ERROR_SYSTEM_ERROR;
    }
    return TK_SUCCESS;
}

static void cleanup_conversation_memory(conversation_memory_t *mem)
{
    if (!mem) return;
    pthread_mutex_lock(&mem->mutex);
    for (size_t i = 0; i < mem->turn_count; ++i) {
        free(mem->turns[i].content);
    }
    free(mem->turns);
    mem->turns = NULL;
    mem->capacity = mem->turn_count = mem->next_index = 0;
    pthread_mutex_unlock(&mem->mutex);
    pthread_mutex_destroy(&mem->mutex);
}

/* ---------- add a context item (deep copy) -------------------------------- */
static tk_error_code_t add_context_item_internal(tk_contextual_reasoner_t *r,
                                                const tk_context_item_t *src)
{
    if (!r || !src) return TK_ERROR_INVALID_ARGUMENT;

    context_memory_t *mem = &r->context_memory;
    pthread_mutex_lock(&mem->mutex);

    /* allocate a fresh slot (circular buffer) */
    size_t idx = mem->next_index;
    tk_context_item_t *dst = &mem->items[idx];

    /* free any previous data that might be overwritten */
    free(dst->description);
    free(dst->data);
    memset(dst, 0, sizeof(*dst));

    /* deep‑copy description */
    if (src->description) {
        dst->description = strdup(src->description);
        if (!dst->description) {
            pthread_mutex_unlock(&mem->mutex);
            return TK_ERROR_OUT_OF_MEMORY;
        }
    }

    /* deep‑copy optional binary payload */
    if (src->data && src->data_size) {
        dst->data = malloc(src->data_size);
        if (!dst->data) {
            free(dst->description);
            pthread_mutex_unlock(&mem->mutex);
            return TK_ERROR_OUT_OF_MEMORY;
        }
        memcpy(dst->data, src->data, src->data_size);
        dst->data_size = src->data_size;
    }

    /* copy scalar fields */
    dst->timestamp_ns    = src->timestamp_ns;
    dst->type            = src->type;
    dst->priority        = src->priority;
    dst->relevance_score = src->relevance_score;

    /* bookkeeping */
    if (mem->item_count < mem->capacity) mem->item_count++;
    mem->next_index = (mem->next_index + 1) % mem->capacity;

    /* update global memory usage estimate */
    pthread_mutex_lock(&r->state_mutex);
    r->system_state.total_memory_bytes +=
        sizeof(*dst) +
        (dst->description ? strlen(dst->description) + 1 : 0) +
        dst->data_size;
    pthread_mutex_unlock(&r->state_mutex);

    pthread_mutex_unlock(&mem->mutex);
    return TK_SUCCESS;
}

/* ---------- relevance handling -------------------------------------------- */
static void update_context_relevance_scores(tk_contextual_reasoner_t *r,
                                            uint64_t now_ns)
{
    context_memory_t *mem = &r->context_memory;
    pthread_mutex_lock(&mem->mutex);
    for (size_t i = 0; i < mem->item_count; ++i) {
        tk_context_item_t *it = &mem->items[i];
        it->relevance_score = calculate_relevance_score(it,
                                                        now_ns,
                                                        r->config.memory_decay_rate);
    }
    pthread_mutex_unlock(&mem->mutex);
}

/* decay = exp(-rate * age_seconds) */
static float calculate_relevance_score(const tk_context_item_t *item,
                                       uint64_t now_ns,
                                       float decay_rate)
{
    if (!item) return 0.0f;
    double age_s = (now_ns - item->timestamp_ns) / 1e9;
    float decay = expf(-decay_rate * (float)age_s);
    return item->relevance_score * decay;
}

/* ---------------------------------------------------------------------- */
static void prune_irrelevant_context(tk_contextual_reasoner_t *r)
{
    context_memory_t *mem = &r->context_memory;
    float threshold = r->config.context_relevance_threshold;

    pthread_mutex_lock(&mem->mutex);
    size_t write = 0;
    for (size_t read = 0; read < mem->item_count; ++read) {
        tk_context_item_t *it = &mem->items[read];
        if (it->relevance_score >= threshold) {
            if (write != read) mem->items[write] = *it;
            ++write;
        } else {
            /* free discarded item */
            free(it->description);
            free(it->data);
        }
    }
    mem->item_count = write;
    mem->next_index = write % mem->capacity;
    pthread_mutex_unlock(&mem->mutex);
}

/* ---------- description generators ---------------------------------------- */
static char *generate_environmental_description(tk_contextual_reasoner_t *r)
{
    pthread_mutex_lock(&r->state_mutex);
    size_t cnt = r->environmental_state.visible_object_count;
    tk_vision_object_t *objs = r->environmental_state.visible_objects;
    pthread_mutex_unlock(&r->state_mutex);

    if (cnt == 0) return strdup("No visible objects");

    /* Build a short textual summary – we limit to 3 objects for brevity */
    size_t limit = cnt < 3 ? cnt : 3;
    char buf[256];
    size_t pos = 0;
    for (size_t i = 0; i < limit; ++i) {
        const tk_vision_object_t *o = &objs[i];
        int n = snprintf(buf + pos, sizeof(buf) - pos,
                         "%s (%.1fm, %.0f%% confidence); ",
                         o->label,
                         o->distance_meters,
                         o->confidence * 100.0f);
        if (n < 0 || (size_t)n >= sizeof(buf) - pos) break;
        pos += (size_t)n;
    }
    if (pos > 2) buf[pos - 2] = '\0';   /* strip trailing "; " */
    return strdup(buf);
}

/* ---------------------------------------------------------------------- */
static char *generate_navigation_description(tk_contextual_reasoner_t *r)
{
    pthread_mutex_lock(&r->state_mutex);
    bool clear = r->navigation_state.has_clear_path;
    float dir = r->navigation_state.clear_path_direction_deg;
    float dist = r->navigation_state.clear_path_distance_m;
    size_t hazard_cnt = r->navigation_state.hazard_count;
    tk_navigation_hazard_t *hazards = r->navigation_state.hazards;
    pthread_mutex_unlock(&r->state_mutex);

    char buf[256];
    if (clear) {
        snprintf(buf, sizeof(buf),
                 "Clear path ahead at %.0f°, %.1fm away. %zu hazards detected.",
                 dir, dist, hazard_cnt);
    } else {
        snprintf(buf, sizeof(buf),
                 "No clear path. %zu hazards detected.", hazard_cnt);
    }
    return strdup(buf);
}

/* ---------------------------------------------------------------------- */
static char *generate_conversation_summary(tk_contextual_reasoner_t *r,
                                           size_t max_turns)
{
    conversation_memory_t *c = &r->conversation_memory;
    pthread_mutex_lock(&c->mutex);

    if (c->turn_count == 0) {
        pthread_mutex_unlock(&c->mutex);
        return strdup("No recent conversation");
    }

    /* we walk backwards from the newest turn */
    size_t to_show = max_turns < c->turn_count ? max_turns : c->turn_count;
    char buf[512];
    size_t pos = 0;
    for (size_t i = 0; i < to_show; ++i) {
        size_t idx = (c->next_index + c->capacity - 1 - i) % c->capacity;
        const tk_conversation_turn_t *t = &c->turns[idx];
        const char *who = t->is_user_input ? "User" : "System";
        int n = snprintf(buf + pos, sizeof(buf) - pos,
                         "%s: \"%s\"; ", who, t->content);
        if (n < 0 || (size_t)n >= sizeof(buf) - pos) break;
        pos += (size_t)n;
    }
    if (pos > 2) buf[pos - 2] = '\0';   /* strip trailing "; " */
    pthread_mutex_unlock(&c->mutex);
    return strdup(buf);
}

/* ---------- time helper --------------------------------------------------- */
static uint64_t get_current_time_ns(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
}

/* -------------------------------------------------------------------------- */
/* Optional: register a critical‑event callback (not part of the public API) */
/* -------------------------------------------------------------------------- */
tk_error_code_t tk_contextual_reasoner_set_critical_event_cb(
        tk_contextual_reasoner_t *reasoner,
        tk_critical_event_cb_t cb,
        void *user_data)
{
    if (!reasoner) return TK_ERROR_INVALID_ARGUMENT;
    pthread_mutex_lock(&reasoner->state_mutex);
    reasoner->critical_cb = cb;
    reasoner->critical_user_data = user_data;
    pthread_mutex_unlock(&reasoner->state_mutex);
    return TK_SUCCESS;
}

/* ---------------------------------------------------------------------- */

tk_error_code_t tk_contextual_reasoner_update_motion_context(
    tk_contextual_reasoner_t* reasoner,
    const tk_world_state_t* world_state)
{
    if (!reasoner || !world_state) {
        return TK_ERROR_INVALID_ARGUMENT;
    }

    pthread_mutex_lock(&reasoner->state_mutex);

    tk_motion_state_e old_state = reasoner->navigation_state.motion_state;
    reasoner->navigation_state.motion_state = world_state->motion_state;

    pthread_mutex_unlock(&reasoner->state_mutex);

    // If the state changed, add a context item to log it.
    if (old_state != world_state->motion_state) {
        char description[128];
        const char* state_str = "UNKNOWN";
        switch(world_state->motion_state) {
            case TK_MOTION_STATE_STATIONARY: state_str = "User is now stationary"; break;
            case TK_MOTION_STATE_WALKING:    state_str = "User started walking"; break;
            case TK_MOTION_STATE_RUNNING:    state_str = "User started running"; break;
            case TK_MOTION_STATE_FALLING:    state_str = "Fall detected!"; break;
            default: break;
        }
        snprintf(description, sizeof(description), "%s", state_str);

        tk_contextual_reasoner_add_context_item(reasoner,
            TK_CONTEXT_TYPE_USER_STATE,
            TK_CONTEXT_PRIORITY_MEDIUM,
            description,
            NULL, 0);
    }

    return TK_SUCCESS;
}
