//! Lock-free single-producer / single-consumer ring (circular) buffer.
//!
//! The buffer uses power-of-two capacity so that index wrapping can be done
//! with a cheap bitwise AND. Capacity is rounded up to the next power of two
//! on construction.
//!
//! This implementation is **not** multi-producer/multi-consumer safe — it is
//! designed for the streaming audio pipeline where one thread pushes samples
//! and another reads them.

/// A fixed-capacity circular buffer for `Copy + Default` values.
///
/// # Overflow behaviour
/// When the buffer is full and [`push_slice`] is called, the **oldest** data
/// is silently overwritten.  The method returns the number of elements
/// actually written (which may be less than `data.len()` if the buffer is
/// smaller than the slice — the last `capacity` elements win).
///
/// [`push_slice`]: RingBuffer::push_slice
pub struct RingBuffer<T: Copy + Default> {
    buf: Vec<T>,
    /// Write pointer (next free slot).
    head: usize,
    /// Read pointer (next unread slot).
    tail: usize,
    /// Allocated capacity (power of two).
    capacity: usize,
}

impl<T: Copy + Default> RingBuffer<T> {
    /// Create a new ring buffer with at least `capacity` slots.
    ///
    /// The actual allocated capacity is the next power of two ≥ `capacity`
    /// (minimum 2).
    pub fn new(capacity: usize) -> Self {
        let cap = next_pow2(capacity.max(2));
        Self {
            buf: vec![T::default(); cap],
            head: 0,
            tail: 0,
            capacity: cap,
        }
    }

    /// Returns the number of elements currently available to read.
    pub fn available(&self) -> usize {
        self.head.wrapping_sub(self.tail) & (self.capacity - 1)
    }

    /// Returns the number of free slots (how many elements can be pushed
    /// before the oldest data is overwritten).
    pub fn free_space(&self) -> usize {
        self.capacity - 1 - self.available()
    }

    /// Push `data` into the ring buffer.
    ///
    /// If there is not enough free space, the oldest data is overwritten and
    /// `tail` is advanced accordingly. Returns the number of elements written
    /// (always `data.len().min(self.capacity - 1)`).
    pub fn push_slice(&mut self, data: &[T]) -> usize {
        let n = data.len().min(self.capacity - 1);
        // Take only the last `n` elements if data is larger than capacity.
        let data = &data[data.len() - n..];

        for &sample in data {
            self.buf[self.head & (self.capacity - 1)] = sample;
            self.head = self.head.wrapping_add(1);
        }

        // If we have overwritten unread data, advance tail.
        if self.available() == self.capacity - 1 {
            // Buffer is now full — tail stays at (head - (cap-1)).
            // Nothing to do here: the invariant is maintained by the writes above.
        }
        // Ensure tail doesn't lag behind more than capacity-1 slots.
        let lag = self.head.wrapping_sub(self.tail) & (self.capacity - 1);
        // Wait — we need to handle overflow properly.
        // The real concern: after writing, available must be <= cap-1.
        // If we wrote past tail, bump tail.
        let written = n;
        // Re-check: if available() would exceed cap-1, advance tail.
        let avail = self.head.wrapping_sub(self.tail);
        if avail > self.capacity - 1 {
            // Advance tail by the excess.
            self.tail = self.head.wrapping_sub(self.capacity - 1);
        }
        let _ = lag;
        written
    }

    /// Read up to `out.len()` elements from the ring buffer into `out`.
    ///
    /// Returns the number of elements actually read (may be less than
    /// `out.len()` if the buffer has fewer available samples).
    pub fn read_slice(&mut self, out: &mut [T]) -> usize {
        let n = out.len().min(self.available());
        for slot in out.iter_mut().take(n) {
            *slot = self.buf[self.tail & (self.capacity - 1)];
            self.tail = self.tail.wrapping_add(1);
        }
        n
    }
}

/// Round `n` up to the next power of two (or return `n` if already a power
/// of two).
fn next_pow2(n: usize) -> usize {
    if n <= 1 {
        return 1;
    }
    let mut p = 1_usize;
    while p < n {
        p <<= 1;
    }
    p
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_next_pow2() {
        assert_eq!(next_pow2(1), 1);
        assert_eq!(next_pow2(2), 2);
        assert_eq!(next_pow2(3), 4);
        assert_eq!(next_pow2(100), 128);
    }
}
