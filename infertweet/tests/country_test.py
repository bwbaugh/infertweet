# Copyright (C) 2013 Wesley Baugh
from infertweet import country


def test_estimated_country_original():
    failed = []
    for code, latitude, longitude in country.COUNTRIES:
        result = country.estimated_country(float(latitude), float(longitude))
        if result != code:
            failed.append(code)
    assert not failed


def test_estimated_country_samples():
    assert 'US' == country.estimated_country(38.058576, -97.3472717)
    assert 'NL' == country.estimated_country(51.658927, 5.611267)
