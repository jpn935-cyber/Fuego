file(GLOB_RECURSE SOURCES
    "${CMAKE_CURRENT_LIST_DIR}/../src/*.cpp"
    "${CMAKE_CURRENT_LIST_DIR}/*.cpp"
)

idf_component_register(
    SRCS ${SOURCES}
    INCLUDE_DIRS
        "../include"
)