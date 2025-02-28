struct Account {
    id: Field,
    balance: u32,
}
type Action = u8;
global ACTION_WITHDRAW: u8 = 0;
global ACTION_DEPOSIT: u8 = 1;

struct Transaction {
    account_id: Field,
    sum: u32,
    action: Action,
}

mod annotation_helpers {
    use super::Account;
    use super::Transaction;
    use super::ACTION_WITHDRAW;
    #[requires(account.balance > transaction.sum)]
    #[requires(account.balance < 1000000)]
    fn apply_transaction(account: Account, transaction: Transaction) -> u32 {
        if transaction.action == ACTION_WITHDRAW {
            account.balance - transaction.sum
        } else {
            account.balance + transaction.sum
        }
    }
}
#[requires(forall(|i, j| (0 <= i) & (i < 2) & (0 <= j) & (j < 2) ==>
         (transactions[j].account_id == accounts[i].id ==> param.0[i].balance == annotation_helpers::apply_transaction(accounts[i], transactions[j])) ))]
fn main(
    mut accounts: [Account; 2],
    transactions: [Transaction; 2],
    param: ([Account; 2], [bool; 2])
) -> pub ([Account; 2], [bool; 2]) {
    let mut transactions_success_status: [bool; 2] = [false; 2];
    (accounts, transactions_success_status)
}

