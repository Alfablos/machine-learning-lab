use mimalloc::MiMalloc;
use std::process;

use polars::{
  prelude::LazyCsvReader
};
use polars::datatypes::DataType;
use polars::error::PolarsResult;
use polars::io::RowIndex;
use polars::prelude::{col, lit, LazyFileListReader, LazyFrame, PlPath, ReshapeDimension};
use regex::Regex;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;


fn main() -> Result<(), Box<dyn std::error::Error>> {
  let mut lazy_dataframe: LazyFrame = LazyCsvReader::new(PlPath::new("./car-sales.csv"))
      .with_has_header(true)
      .with_row_index(Some(RowIndex { name: "id".into(), offset: 0 }))
      .finish()?;

  // Adjusting columns: names in lowercase, odometer column with a better naming, prices as floats
  lazy_dataframe = lazy_dataframe
      .select([
        col("id"),
        // Lowercase for values and column names
        col("Make").str().to_lowercase().name().to_lowercase(),
                col("Colour").str().to_lowercase().name().to_lowercase(),
        // col("Doors").name().to_lowercase(),          // 'doors' WILL NOT BE READ from the CSV, will NOT BE LOADED IN MEMORY
        col("Odometer (KM)").name().map(|_name| PolarsResult::Ok("odometer_km".into())),
        col("Price")
          .str().replace_all(lit(r"\$|,"), lit(""), false)
          .cast(DataType::Float64)
          .name().to_lowercase()
        ]);

    lazy_dataframe = lazy_dataframe.filter(col("price").gt(6000.00));

  println!("{}", lazy_dataframe.collect()?);

  process::exit(0);

  // let filter_task = lazy_dataframe
  //     .drop_nans(None)
  //     .select(
  //       col("id"),
  //       // Lowercase for values and column names
  //       col('Make', 'Colour').str.to_lowercase().name.to_lowercase(),
  //   pl.col('Doors').name.to_lowercase(),
  //   pl.col('Odometer (KM)'),
  //          pl.col('Price')
  //              .str.replace_all(r"\$|,|...$", '')
  //              .str.to_integer(base=10)
  //              .name.to_lowercase()
  //   );

  Ok(())
}