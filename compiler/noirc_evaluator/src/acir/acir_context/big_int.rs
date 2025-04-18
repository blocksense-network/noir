use acvm::acir::AcirField;
use num_bigint::BigUint;

/// Represents a bigint value in the form (id, modulus) where
///     id is the identifier of the big integer number, and
///     modulus is the identifier of the big integer size
#[derive(Default, Clone, Copy, Debug)]
pub(super) struct BigIntId {
    pub(super) bigint_id: u32,
    pub(super) modulus_id: u32,
}

impl BigIntId {
    pub(super) fn bigint_id<F: From<u128>>(&self) -> F {
        F::from(self.bigint_id as u128)
    }

    pub(super) fn modulus_id<F: From<u128>>(&self) -> F {
        F::from(self.modulus_id as u128)
    }
}

/// BigIntContext is used to generate identifiers for big integers and their modulus
#[derive(Default, Debug)]
pub(super) struct BigIntContext {
    modulus: Vec<BigUint>,
    big_integers: Vec<BigIntId>,
}

impl BigIntContext {
    /// Creates a new BigIntId for the given modulus identifier and returns it.
    pub(super) fn new_big_int<F: AcirField>(&mut self, modulus_id: F) -> BigIntId {
        let id = self.big_integers.len() as u32;
        let result = BigIntId { bigint_id: id, modulus_id: modulus_id.to_u128() as u32 };
        self.big_integers.push(result);
        result
    }

    /// Returns the modulus corresponding to the given modulus index
    pub(super) fn modulus<F: AcirField>(&self, idx: F) -> BigUint {
        self.modulus[idx.to_u128() as usize].clone()
    }

    /// Returns the BigIntId corresponding to the given identifier
    pub(super) fn get<F: AcirField>(&self, id: F) -> BigIntId {
        self.big_integers[id.to_u128() as usize]
    }

    /// Adds a modulus to the context (if it is not already present)
    pub(super) fn get_or_insert_modulus(&mut self, modulus: BigUint) -> u32 {
        if let Some(pos) = self.modulus.iter().position(|x| x == &modulus) {
            return pos as u32;
        }
        self.modulus.push(modulus);
        (self.modulus.len() - 1) as u32
    }
}
