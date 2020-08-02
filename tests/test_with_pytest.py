import pytest
import irrigation30 as irr


def test_input_types():
    with pytest.raises(ValueError) as exception:
        model = irr.Irrigation30(center_lat="f", center_lon=77.0,
                                 edge_len=0.005, num_clusters=2)

    with pytest.raises(ValueError) as exception:
        model = irr.Irrigation30(center_lat=44.0, center_lon="f",
                                 edge_len=0.005, num_clusters=2)

    with pytest.raises(ValueError) as exception:
        model = irr.Irrigation30(center_lat=44.0, center_lon=77.0,
                                 edge_len="f", num_clusters=2)

    with pytest.raises(ValueError) as exception:
        model = irr.Irrigation30(center_lat=44.0, center_lon=77.0,
                                 edge_len=0.05, num_clusters="f")

    with pytest.raises(ValueError) as exception:
        model = irr.Irrigation30(center_lat=44.0, center_lon=77.0,
                                 edge_len=0.05, num_clusters=2, year="f")

    with pytest.raises(ValueError) as exception:
        model = irr.Irrigation30(center_lat=44.0, center_lon=77.0,
                                 edge_len=0.05, num_clusters=2, year=2018.2)

    with pytest.raises(ValueError) as exception:
        model = irr.Irrigation30(center_lat=44.0, center_lon=77.0,
                                 edge_len=0.05, num_clusters=2, year=2018, base_asset_directory=3)


def test_input_ranges():
    with pytest.raises(ValueError) as exception:
        model = irr.Irrigation30(center_lat=200.0, center_lon=77.0,
                                 edge_len=0.05, num_clusters=2, year=2018)

    with pytest.raises(ValueError) as exception:
        model = irr.Irrigation30(center_lat=44.0, center_lon=200.0,
                                 edge_len=0.05, num_clusters=2, year=2018)

    with pytest.raises(ValueError) as exception:
        model = irr.Irrigation30(center_lat=44.0, center_lon=200.0,
                                 edge_len=2.0, num_clusters=2, year=2018)

    with pytest.raises(ValueError) as exception:
        model = irr.Irrigation30(center_lat=44.0, center_lon=200.0,
                                 edge_len=0.05, num_clusters=12, year=2018)

    with pytest.raises(ValueError) as exception:
        model = irr.Irrigation30(center_lat=44.0, center_lon=200.0,
                                 edge_len=0.05, num_clusters=2, year=2030)
