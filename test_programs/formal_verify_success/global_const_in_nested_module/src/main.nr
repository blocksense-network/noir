use fv_std::U8_MAX;

#['ensures(bar::qux::x == result)]
fn main(y: u32) -> pub u32 {
  (U8_MAX - 250) as u32
}

mod bar {
   mod qux {
    global x: u32 = 5;
  }
}
