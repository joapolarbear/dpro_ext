from typing import Any


class A:
    def __init__(self):
        pass

    def test1(self):
        print("I am A")
        
    def overide(self):
        print("to be override")
        
class B:
    def __init__(self, a):
        self.a = a
    
    def overide(self):
        print("Sucessfully override")
    
    # def __getattribute__(self, __name: str) -> Any:
    #     if __name in self.__dict__:
    #         return self.__dict__[__name]
    #     return self.a.__dict__[__name]
    
a = A()
b = B(a)
b.overide()
b.test1()