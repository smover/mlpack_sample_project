# build the library for the sub module

include_directories(.)

add_library(submodule SHARED
  myclass.cpp
)
