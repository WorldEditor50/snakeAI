cmake_minimum_required(VERSION 3.16)

project(SimpleRL LANGUAGES CXX)

set(CMAKE_INCLUDE_CURRENT_DIR ON)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Single-config generators (Ninja / Makefiles) default to Release so a plain
# `cmake -S . -B build` does not silently produce an unoptimised build.
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

option(SNAKEAI_BUILD_TESTS "Build the test executables" ON)
option(SNAKEAI_USE_PCH
       "Precompile rl/tensor.hpp + rl/layer.h for the RL library (CMake >= 3.16)"
       OFF)

# Enables `ctest` to see the test executables declared at the bottom of this
# file. Harmless when SNAKEAI_BUILD_TESTS is OFF.
enable_testing()

# ============================================================================
# Build organisation / why it is laid out this way
# ----------------------------------------------------------------------------
# The RL layer (rl/) is pure C++ and never touches Qt. The previous build file
# globbed rl/*.h + rl/*.hpp + rl/*.cpp together, appended them to every
# executable, and had CMAKE_AUTOMOC ON globally. Consequences:
#   * the same 16 translation units were recompiled by all 11 executables
#     (176 compiler invocations instead of 16);
#   * AUTOMOC ran on all 11 targets, including the 10 plain-C++ test binaries,
#     producing 11 *_autogen directories and re-moc'ing every RL header 11x;
#   * headers were listed as if they were sources.
# So: one STATIC library for the RL core (compiled once, Qt-free, no code
# generators), and Qt's code generators enabled only on the GUI target.
# ============================================================================

find_package(QT NAMES Qt6 Qt5 COMPONENTS Widgets REQUIRED)
find_package(Qt${QT_VERSION_MAJOR} COMPONENTS Widgets REQUIRED)

set(SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR})

# ============================================================ RL core (no Qt)
# Explicit source lists instead of file(GLOB): deterministic, no re-glob on
# configure, and headers are not passed to the compiler as translation units.
# Add new rl/*.cpp here when you add a file.
set(RL_SOURCES
    rl/bcq.cpp
    rl/convdqn.cpp
    rl/convpg.cpp
    rl/ddpg.cpp
    rl/dpg.cpp
    rl/dqn.cpp
    rl/drpg.cpp
    rl/gru.cpp
    rl/lstm.cpp
    rl/mamba.cpp
    rl/mpg.cpp
    rl/ppo.cpp
    rl/sac.cpp
    rl/ssm.cpp
    rl/trpo.cpp
    rl/util.cpp
)
add_library(RL_CORE STATIC ${RL_SOURCES})

target_include_directories(RL_CORE PUBLIC ${SRC_DIR})
set_target_properties(RL_CORE PROPERTIES
    AUTOMOC OFF
    AUTOUIC OFF
    AUTORCC OFF)

# ---------------------------------------------------------------------------
# Workaround: make header changes actually trigger rebuilds.
#
# On this machine MSVC emits its /showIncludes note in the localized form
# ("注意: 包含文件:") while the prefix CMake recorded during compiler detection is
# a different (mis-decoded) byte sequence, so ninja matches nothing and stores
# NO header dependencies. Consequence: editing rl/tensor.hpp or rl/layer.h
# rebuilt nothing, incremental builds silently produced STALE binaries, and a
# crash fix appeared to have no effect while an AddressSanitizer build of the
# same source was clean. Declaring the headers as explicit object dependencies
# of every source keeps incremental builds correct regardless of locale.
# ---------------------------------------------------------------------------
file(GLOB RL_HEADER_DEPS
    ${SRC_DIR}/rl/*.h
    ${SRC_DIR}/rl/*.hpp)
set_property(SOURCE ${RL_SOURCES} APPEND PROPERTY OBJECT_DEPENDS ${RL_HEADER_DEPS})

target_compile_options(RL_CORE PUBLIC
    $<$<CXX_COMPILER_ID:MSVC>:/bigobj>)

# Detect Ninja once, explicitly, so the MSVC condition below cannot be misread
# by CMake's if() operator precedence.
set(SNAKEAI_GENERATOR_IS_NINJA FALSE)
if(CMAKE_GENERATOR MATCHES "Ninja")
    set(SNAKEAI_GENERATOR_IS_NINJA TRUE)
endif()

if(MSVC AND NOT SNAKEAI_GENERATOR_IS_NINJA)
    # /MP parallelises the TUs *within* one target. With Ninja that is
    # counterproductive (Ninja already runs many compilers in parallel), so it
    # is only added for MSBuild-style generators.
    target_compile_options(RL_CORE PUBLIC /MP)
endif()

if(SNAKEAI_USE_PCH)
    # Typically another 20-40% off the RL library. Kept opt-in because it must
    # be re-validated against the toolchain before becoming the default.
    target_precompile_headers(RL_CORE PRIVATE
        rl/tensor.hpp
        rl/layer.h)
endif()

# =============================================================== GUI program
set(APP_SOURCES
    main.cpp
    agent.cpp
    axis.cpp
    common.cpp
    environment.cpp
    gamewidget.cpp
    genetic.cpp
    mainwindow.cpp
    mainwindow.ui
    snake.cpp
)
add_executable(SimpleRL ${APP_SOURCES})

# Same header-dependency workaround as for the RL library above (the app also
# includes the top-level headers).
file(GLOB APP_HEADER_DEPS
    ${SRC_DIR}/*.h
    ${SRC_DIR}/rl/*.h
    ${SRC_DIR}/rl/*.hpp)
set_property(SOURCE ${APP_SOURCES} APPEND PROPERTY OBJECT_DEPENDS ${APP_HEADER_DEPS})

# Qt's code generators are needed by this target only.
set_target_properties(SimpleRL PROPERTIES
    AUTOMOC ON
    AUTOUIC ON
    AUTORCC ON)

target_link_libraries(SimpleRL PRIVATE
    RL_CORE
    Qt${QT_VERSION_MAJOR}::Widgets)

target_compile_options(SimpleRL PRIVATE
    $<$<CXX_COMPILER_ID:MSVC>:/bigobj>)

if(SNAKEAI_USE_PCH)
    target_precompile_headers(SimpleRL PRIVATE
        rl/tensor.hpp
        rl/layer.h)
endif()

# ===================================================================== tests
if(SNAKEAI_BUILD_TESTS)
    # These are plain C++ programs that only include rl/ headers, so they link
    # the shared RL library and skip Qt entirely.
    #
    # The core RL test is built as `test_rl` from test/test.cpp: once
    # enable_testing() is active, CTest reserves the target name `test` for
    # itself, and add_executable(test ...) is a hard configure error.
    foreach(t IN ITEMS
            test_rl test_dqn test_lstm test_ssm test_mamba
            test_moe test_mpg test_pg test_sac test_trpo test_gradcheck test_convpg
            test_convdqn)
        if(t STREQUAL "test_rl")
            set(_src test/test.cpp)
        else()
            set(_src test/${t}.cpp)
        endif()
        add_executable(${t} ${_src})
        target_link_libraries(${t} PRIVATE RL_CORE)
        set_target_properties(${t} PROPERTIES
            AUTOMOC OFF
            AUTOUIC OFF
            AUTORCC OFF)
        target_compile_options(${t} PRIVATE
            $<$<CXX_COMPILER_ID:MSVC>:/bigobj>)
        # Register with CTest so the whole suite can be run with
        # `ctest --test-dir build/verify`. Without this the test executables
        # were built but `ctest` reported "No tests were found".
        add_test(NAME ${t} COMMAND ${t})
        # These are stochastic training tests; the default 1500s is generous but
        # an explicit limit keeps a hung run from blocking the suite.
        set_tests_properties(${t} PROPERTIES TIMEOUT 600)
    endforeach()

    # Regression test for the reward plot. This is the only test that touches
    # Qt: AxisWidget is a QWidget, so it needs AUTOMOC and the Widgets module.
    # It runs offscreen, so no window system is required.
    add_executable(test_axis test/test_axis.cpp axis.cpp axis.h)
    target_link_libraries(test_axis PRIVATE Qt${QT_VERSION_MAJOR}::Widgets)
    target_include_directories(test_axis PRIVATE ${SRC_DIR})
    set_target_properties(test_axis PROPERTIES
        AUTOMOC ON AUTOUIC OFF AUTORCC OFF)
    add_test(NAME test_axis COMMAND test_axis)
    set_tests_properties(test_axis PROPERTIES
        TIMEOUT 120
        ENVIRONMENT "QT_QPA_PLATFORM=offscreen")

    # End-to-end benchmark for the agents, driven through the REAL Environment
    # exactly as the GUI does. Deliberately NOT registered with CTest: one run
    # takes tens of seconds per agent, and it prints a learning curve rather
    # than a pass/fail. Build it and run
    #   bench_agent_game <agent> <steps> <phases>
    # e.g. `bench_agent_game convdqn 2000 10`. It reports, per phase, targets
    # eaten, the share of moves that reduced the distance to the target
    # (approach%), mean snake length and ms/step. Reference points measured on
    # this machine over 1200 steps: astar 99%, rand ~76%, dqn 62->98%,
    # dpg/drpg/mpg/convpg 84->96%.
    add_executable(bench_agent_game
        test/bench_agent_game.cpp
        agent.cpp
        common.cpp
        environment.cpp
        snake.cpp)
    target_link_libraries(bench_agent_game PRIVATE RL_CORE)
    target_include_directories(bench_agent_game PRIVATE ${SRC_DIR})
    target_compile_options(bench_agent_game PRIVATE
        $<$<CXX_COMPILER_ID:MSVC>:/bigobj>)
    set_target_properties(bench_agent_game PROPERTIES
        AUTOMOC OFF AUTOUIC OFF AUTORCC OFF)
endif()
