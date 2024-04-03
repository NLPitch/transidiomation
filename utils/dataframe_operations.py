import polars as pl

def group_by_aggregate(pl_data, group_by_factor:str):

    return pl_data.group_by(group_by_aggregate).agg([pl.all()])

def group_by_and_random_representation(pl_data, group_by_factor:str):
    pl_data = pl_data.select(
        pl.col(group_by_factor),
        remaining_data = pl.struct(pl.exclude(group_by_factor))
    )

    pl_data = pl_data.group_by(group_by_factor).agg([pl.col('remaining_data')])
    pl_data = pl_data.with_columns(
        pl.col('remaining_data').list.get(0)
    ).unnest('remaining_data')

    return pl_data

def random_row_seleection(pl_data, size:int):

    return pl_data.sample(n=size)