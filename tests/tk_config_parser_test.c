#define _DEFAULT_SOURCE
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include "../src/internal_tools/tk_config_parser.h"

// Helper function to create a temporary config file
static char* create_temp_config_file(const char* content) {
    char* filepath = tmpnam(NULL);
    FILE* file = fopen(filepath, "w");
    if (!file) {
        perror("fopen");
        return NULL;
    }
    fputs(content, file);
    fclose(file);
    return filepath;
}

static void test_get_bool_true_values() {
    printf("Running test: test_get_bool_true_values\n");
    const char* content =
        "key_true=true\n"
        "key_yes=yes\n"
        "key_on=on\n"
        "key_1=1\n";
    char* filepath = create_temp_config_file(content);
    assert(filepath != NULL);

    tk_config_t* config = NULL;
    tk_error_code_t err = tk_config_create(&config);
    assert(err == TK_SUCCESS);

    err = tk_config_load_from_file(config, filepath);
    assert(err == TK_SUCCESS);

    assert(tk_config_get_bool(config, "key_true", false) == true);
    assert(tk_config_get_bool(config, "key_yes", false) == true);
    assert(tk_config_get_bool(config, "key_on", false) == true);
    assert(tk_config_get_bool(config, "key_1", false) == true);

    tk_config_destroy(&config);
    remove(filepath);
    printf("Test passed: test_get_bool_true_values\n");
}

static void test_get_bool_false_values() {
    printf("Running test: test_get_bool_false_values\n");
    const char* content =
        "key_false=false\n"
        "key_no=no\n"
        "key_off=off\n"
        "key_0=0\n"
        "key_other=other\n";
    char* filepath = create_temp_config_file(content);
    assert(filepath != NULL);

    tk_config_t* config = NULL;
    tk_error_code_t err = tk_config_create(&config);
    assert(err == TK_SUCCESS);

    err = tk_config_load_from_file(config, filepath);
    assert(err == TK_SUCCESS);

    assert(tk_config_get_bool(config, "key_false", true) == false);
    assert(tk_config_get_bool(config, "key_no", true) == false);
    assert(tk_config_get_bool(config, "key_off", true) == false);
    assert(tk_config_get_bool(config, "key_0", true) == false);
    assert(tk_config_get_bool(config, "key_other", true) == true);
    assert(tk_config_get_bool(config, "key_other", false) == false);

    tk_config_destroy(&config);
    remove(filepath);
    printf("Test passed: test_get_bool_false_values\n");
}

static void test_get_bool_default_values() {
    printf("Running test: test_get_bool_default_values\n");
    const char* content = "some_key=some_value\n";
    char* filepath = create_temp_config_file(content);
    assert(filepath != NULL);

    tk_config_t* config = NULL;
    tk_error_code_t err = tk_config_create(&config);
    assert(err == TK_SUCCESS);

    err = tk_config_load_from_file(config, filepath);
    assert(err == TK_SUCCESS);

    assert(tk_config_get_bool(config, "non_existent_key", true) == true);
    assert(tk_config_get_bool(config, "non_existent_key", false) == false);

    tk_config_destroy(&config);
    remove(filepath);
    printf("Test passed: test_get_bool_default_values\n");
}

int main() {
    test_get_bool_true_values();
    test_get_bool_false_values();
    test_get_bool_default_values();
    printf("All config parser tests passed!\n");
    return 0;
}
