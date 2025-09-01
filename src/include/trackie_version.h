/*
* Copyright (C) 2025 Pedro Henrique / phdev13
*
* File: trackie_version.h
*
* This header provides macros for the TrackieLLM library version. These
* macros allow for compile-time checks and runtime reporting of the library
* version. The versioning scheme follows Semantic Versioning 2.0.0 (semver.org).
*
* TRACKIELLM_VERSION_MAJOR: Incremented for incompatible API changes.
* TRACKIELLM_VERSION_MINOR: Incremented for adding functionality in a
*                           backwards-compatible manner.
* TRACKIELLM_VERSION_PATCH: Incremented for backwards-compatible bug fixes.
*
* TRACKIELLM_VERSION_STRING provides a human-readable string literal.
* TRACKIELLM_VERSION_HEX provides a single integer for easy numerical comparisons.
*
* SPDX-License-Identifier:
*/

#ifndef TRACKIELLM_TRACKIE_VERSION_H
#define TRACKIELLM_TRACKIE_VERSION_H

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @file trackie_version.h
 * @brief TrackieLLM library version definitions.
 */

/**
 * @def TRACKIELLM_VERSION_MAJOR
 * @brief Major version component.
 *
 * Changed for incompatible API changes.
 */
#define TRACKIELLM_VERSION_MAJOR 0

/**
 * @def TRACKIELLM_VERSION_MINOR
 * @brief Minor version component.
 *
 * Changed for new, backwards-compatible functionality.
 */
#define TRACKIELLM_VERSION_MINOR 1

/**
 * @def TRACKIELLM_VERSION_PATCH
 * @brief Patch version component.
 *
 * Changed for backwards-compatible bug fixes.
 */
#define TRACKIELLM_VERSION_PATCH 0

/**
 * @def TRACKIELLM_VERSION_STRING
 * @brief A string literal representing the full version.
 *
 * Format is "major.minor.patch".
 */
#define TRACKIELLM_VERSION_STRING "0.1.0"

/**
 * @def TRACKIELLM_VERSION_HEX
 * @brief A hexadecimal integer representation of the version.
 *
 * This format allows for easy programmatic version comparisons.
 * The format is 0xMMNNPP00 where MM is major, NN is minor, and PP is patch.
 * For example, version 1.2.3 would be 0x01020300.
 */
#define TRACKIELLM_VERSION_HEX ((TRACKIELLM_VERSION_MAJOR << 24) | \
                                (TRACKIELLM_VERSION_MINOR << 16) | \
                                (TRACKIELLM_VERSION_PATCH << 8))

/**
 * @brief Retrieves the major version number of the library.
 *
 * @return The major version number.
 */
int trackiellm_version_major(void);

/**
 * @brief Retrieves the minor version number of the library.
 *
 * @return The minor version number.
 */
int trackiellm_version_minor(void);

/**
 * @brief Retrieves the patch version number of the library.
 *
 * @return The patch version number.
 */
int trackiellm_version_patch(void);

/**
 * @brief Retrieves the full version string of the library.
 *
 * @return A constant string representing the version (e.g., "0.1.0").
 */
const char* trackiellm_version_string(void);

/**
 * @brief Retrieves the version as a single hexadecimal number.
 *
 * This is useful for quick and efficient runtime version checks.
 *
 * @return An integer representing the version in 0xMMNNPP00 format.
 */
unsigned int trackiellm_version_hex(void);


#ifdef __cplusplus
} // extern "C"
#endif

#endif // TRACKIELLM_TRACKIE_VERSION_H