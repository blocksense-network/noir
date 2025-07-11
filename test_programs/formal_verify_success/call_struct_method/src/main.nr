struct A {
  id: Field,
  balance: u32
}

impl A {
    #['ensures(result.balance == self.balance / 2)]
    fn foo(mut self) -> Self {
        self.balance = self.balance / 2;
        self
    }
}

fn main(x: A) -> pub A {
    x.foo()
}

