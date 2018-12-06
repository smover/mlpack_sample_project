#include "submodule/submodule.hpp"

int main () {
  submodule::MyClass myclass = submodule::MyClass();

  myclass.flip();

  return 0 == (myclass.isEnabled());
}
