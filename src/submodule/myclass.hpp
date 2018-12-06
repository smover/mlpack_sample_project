/**
 * Header for the MyClass data structure
 */

/* Include the header only once */
#ifndef MYCLASS_H
#define MYCLASS_H

namespace submodule {
  class MyClass {
  public:
    MyClass() : enabled(false) {};

    bool isEnabled() const {
      return enabled;
    };

    void flip();

  private:
    bool enabled;
  };
}

#endif
