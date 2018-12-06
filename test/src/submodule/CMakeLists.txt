# add the executable to test the submodule

add_executable(testsubmodule ${PROJECT_SOURCE_DIR}/test/src/submodule/test.cpp)
target_link_libraries(testsubmodule
  submodule)

add_test(submoduleflips testsubmodule)


