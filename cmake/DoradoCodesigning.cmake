# Helper script to codesign dorado as part of CPack packaging.
#
# Note that this script is intended to run after CPack install phase but before the build.
#

# Grab the identity from the environment
set(DORADO_CODESIGN_IDENTITY "$ENV{APPLE_CODESIGN_IDENTITY}")
if ("${DORADO_CODESIGN_IDENTITY}" STREQUAL "")
    message(FATAL_ERROR "Trying to sign a build without setting a codesign identity (APPLE_CODESIGN_IDENTITY envvar)")
endif()

# Sign it
message(STATUS "Signing dorado")
execute_process(
    COMMAND
        codesign
        --sign "${DORADO_CODESIGN_IDENTITY}"
        --timestamp
        --options=runtime
        -vvvv
        "${CPACK_TEMPORARY_INSTALL_DIRECTORY}/bin/dorado"
    RESULT_VARIABLE
        SIGNING_RESULT
)
if (NOT ${SIGNING_RESULT} EQUAL 0)
    message(FATAL_ERROR "Signing failed")
endif()
