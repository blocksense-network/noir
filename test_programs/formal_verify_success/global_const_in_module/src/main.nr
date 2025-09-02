use fv_std;

#['ensures(bar::x == fv_std::U8_MIN)]
fn main() {
}

mod bar {
  global x: u32 = 0;
}
