use ndarray::{ArrayBase, OwnedRepr, Ix1, Ix2, array, Array, Ix3, Array2, Axis};
use polars::prelude::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
  /*
  sales_by_week = np.array([[2, 7, 1],
                          [9, 4, 16],
                          [11, 14, 18],
                          [13, 13, 16],
                          [15,18,9]])

  prices = np.array([10, 8, 12])

  earnings_by_product = sales_by_week.dot(prices)


  sales_by_week = pd.DataFrame(data=sales_by_week,
                             index=["Mon", "Tue", "Wed", "Thu", "Fri"],
                             columns=["Almond butter", "Peanut butter", "Cashew btter"])

  sales_by_week["Total ($)"] = earnings_by_product

  */

  let sales_per_week: ArrayBase<OwnedRepr<u16>, Ix2> = array![[2, 7, 1],
                          [9, 4, 16],
                          [11, 14, 18],
                          [13, 13, 16],
                          [15,18,9]];

  let prices: ArrayBase<OwnedRepr<u16>, Ix1> = array![10, 8, 12];
  let prices: ArrayBase<OwnedRepr<u16>, Ix2> = Array::from_shape_vec((3, 1), prices.to_vec())?;

  let earnings_by_product = sales_per_week.dot(&prices);  // u16 because the results cannot be contained in u8

  println!("{}", earnings_by_product);

    let sales_by_week_df = "";

  Ok(())

}

fn array2_to_dataframe(array: &Array2<i32>) -> PolarsResult<DataFrame> {
    let mut series_vec = Vec::new();

    for (col_idx, column) in array.axis_iter(Axis(1)).enumerate() {
        let name = format!("col_{}", col_idx);
        let values = column.to_vec();
        series_vec.push(Series::new((&name).into(), values));
    }

    df!(series_vec)
}