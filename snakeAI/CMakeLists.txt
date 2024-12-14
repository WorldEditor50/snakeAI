cmake_minimum_required(VERSION 3.5)

project(SimpleRL LANGUAGES CXX)

set(CMAKE_INCLUDE_CURRENT_DIR ON)

set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# QtCreator supports the following variables for Android, which are identical to qmake Android variables.
# Check http://doc.qt.io/qt-5/deployment-android.html for more information.
# They need to be set before the find_package(Qt5 ...) call.

find_package(QT NAMES Qt6 Qt5 COMPONENTS Widgets REQUIRED)
find_package(Qt${QT_VERSION_MAJOR} COMPONENTS Widgets REQUIRED)

# src
set(SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR})

file(GLOB SRC_FILES
    ${SRC_DIR}/*.h
    ${SRC_DIR}/*.hpp
    ${SRC_DIR}/*.cpp
    ${SRC_DIR}/*.ui)
# rl
file(GLOB RL_FILES
    ${SRC_DIR}/rl/*.h
    ${SRC_DIR}/rl/*.hpp
    ${SRC_DIR}/rl/*.cpp)
list(APPEND SRC_FILES ${RL_FILES})

# test
file(GLOB TEST_FILES
    ${SRC_DIR}/test/*.h
    ${SRC_DIR}/test/*.cpp)
add_executable(test ${RL_FILES} ${TEST_FILES})

add_executable(SimpleRL ${SRC_FILES})
target_link_libraries(SimpleRL PRIVATE
    Qt${QT_VERSION_MAJOR}::Widgets)

