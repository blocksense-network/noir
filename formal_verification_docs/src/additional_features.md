# Additional Features

Noir FV supports formal verification for **structures**, **tuples**, and **generic functions**, enabling proofs for more complex data types and abstractions.

## Verifying Structs
### Example: Verifying a Student’s Scholarship Eligibility
```rust,ignore
global MIN_GRADE: u8 = 8;
global MAX_GRADE: u8 = 10;

struct Student {
    id: Field,
    grade: u8,
}

// Returns true if the student qualifies for the scholarship 
#[requires(student.grade <= MAX_GRADE)] // Max possible grade
#[ensures(student.grade >= MIN_GRADE ==> result == true)]
#[ensures(student.grade < MIN_GRADE ==> result == false)]
fn main(student: Student) -> pub bool{
    student.grade >= MIN_GRADE
}
```
Noir FV enables writing precise and verifiable specifications for structured data, reducing potential errors and improving code reliability.

## Verifying Traits and Generic Functions

Noir FV supports **traits** and **generic functions**, allowing for modular verification across different implementations.

### Example: Verifying a Banking System
This example defines a `BankAccount` trait and verifies that different account types correctly implement it.
```rust,ignore
trait BankAccount {
  fn get_amount(self) -> u64;
}

struct NamedAccount {
  name: str<10>,
  amount: u32,
}

impl BankAccount for NamedAccount {
  #[ensures(result == self.amount as u64)]
  fn get_amount(self) -> u64 {
    self.amount as u64
  }
}

struct AnonymousAccount {
  encrypted_id: Field,
  amount: u32,
}

impl BankAccount for AnonymousAccount {
  #[ensures(result == self.amount as u64)]
  fn get_amount(self) -> u64 {
    self.amount as u64
  }
}

#[ensures(result == (first_account.amount as u64 + second_account.amount as u64) )]
fn main(first_account: NamedAccount, second_account: AnonymousAccount) -> pub u64 {
  sum_accounts(first_account, second_account)
}

#[ensures(result == first.get_amount() + second.get_amount())]
fn sum_accounts<T, U>(first: T, second: U) -> pub u64 
where T: BankAccount,
      U: BankAccount,
{
  first.get_amount() + second.get_amount()
}
```