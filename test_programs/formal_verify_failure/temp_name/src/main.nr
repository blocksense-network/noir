global U32_MAX: u32 = 4294967295;

#[derive(Eq)]
struct Account {
    id: u32,
    balance: u32,
}

impl Account {
    #[requires(self.balance >= amount)]
    #[ensures((result.balance == self.balance - amount))]
    fn withdraw(mut self, amount: u32) -> Self {
        self.balance -= amount;
        self
    }

    #[requires(self.balance + amount <= U32_MAX)]
    #[ensures((result.balance == self.balance + amount))]
    fn deposit(mut self, amount: u32) -> Self {
        self.balance += amount;
        self
    }
}

type Action = u8;
global ACTION_WITHDRAW: u8 = 0;
global ACTION_DEPOSIT: u8 = 1;

#[derive(Eq)]
struct Transaction {
    account_id: u32,
    sum: u32,
    action: Action,
}

fn will_overflow(arg1: u32, arg2: u32) -> bool {
    if arg1 > U32_MAX - arg2 { true } else { false }
}

fn will_underflow(arg1: u32, arg2: u32) -> bool {
    if arg1 < arg2 { true } else { false }
}

impl Transaction {
    #[requires((self.account_id == account.id) & ((self.action == ACTION_WITHDRAW) | (self.action == ACTION_DEPOSIT)))]
    #[requires((self.action == ACTION_DEPOSIT) ==> (!will_overflow(account.balance, self.sum)))]
    #[requires((self.action == ACTION_WITHDRAW) ==> (!will_underflow(account.balance, self.sum)))]
    #[ensures((self.action == ACTION_DEPOSIT) ==> (result.balance == account.balance + self.sum))]
    #[ensures((self.action == ACTION_WITHDRAW) ==> (result.balance == account.balance - self.sum))]
    fn apply(self, account: Account) -> Account {
        if self.action == ACTION_WITHDRAW {
            account.withdraw(self.sum)
        } else {
            account.deposit(self.sum)
        }
    }
}
// This function is only used in specifications, therefore no annotaions for it are required.
// It returns the newly updated balance
mod annotation_helpers {
    use super::Account;
    use super::Transaction;
    use super::ACTION_WITHDRAW;
    fn apply_transaction(accout: Account, transaction: Transaction) -> u32 {
        if transaction.action == ACTION_WITHDRAW {
            accout.balance - transaction.sum
        } else {
            accout.balance + transaction.sum
        }
    }
}
// Our program can have at most one transaction per account.
#[requires(forall(|i, j| (0 <= i) & (i < j) & (j < 4) ==> transactions[i].account_id != transactions[j].account_id))]
//#[ensures(forall(|i, j| (0 <= i) & (i < 10) & (0 <= j) & (j < 10) ==>
 //         (transactions[j].account_id == accounts[i].id ==> result.0[i].balance == annotation_helpers::apply_transaction(accounts[i], transactions[j])) ))]
// Returns the updated accounts and array of flags that indicates which transactions were successfull
fn main(mut accounts: [Account; 4], transactions: [Transaction; 4]) -> pub ([Account; 4], [bool; 4]) {
    let mut transactions_success_status: [bool; 4] = [false; 4];
    for i in 0..4 {
        for j in 0..4 {
            if transactions[j].account_id == accounts[i].id {
                if ((transactions[j].action == ACTION_DEPOSIT)
                    & (!will_overflow(accounts[i].balance, transactions[j].sum)))
                    | ((transactions[j].action == ACTION_WITHDRAW)
                        & (!will_underflow(accounts[i].balance, transactions[j].sum))) {
                    accounts[i] = transactions[j].apply(accounts[i]);
                    transactions_success_status[j] = true;
                } else {
                    transactions_success_status[j] = false;
                }
            }
        }
    }
    (accounts, transactions_success_status)
}

