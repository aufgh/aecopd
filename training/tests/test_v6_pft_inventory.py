from patient_data.pft_inventory import candidates


def test_ratio_not_fev1_percent_predicted():
    rows = candidates('FEV1/FVC 65%, FEV1 1.2 L (40% predicted), FVC 2.4 L')
    assert [r['metric'] for r in rows] == ['FEV1/FVC', 'FEV1', 'FVC']
    assert rows[0]['quantity_role'] == 'ratio_percent'
    assert rows[1]['quantity_role'] == 'volume_l'


def test_percent_not_assumed_predicted():
    assert candidates('FEV1 40%')[0]['quantity_role'] == 'percent_unspecified'
    assert candidates('FEV1 40% predicted')[0]['quantity_role'] == 'percent_predicted_lexical'


def test_not_borrow_unit_from_next_metric():
    row = candidates('FEV1 0.9, FVC 2.0 L (50% predicted)')[0]
    assert row['quantity_role'] == 'unit_unknown'


def test_unknown_ratio_scale_kept():
    assert candidates('FEV1/FVC .65')[0]['value'] == .65
    assert candidates('FEV1/FVC .65')[0]['quantity_role'] == 'unit_unknown'


def test_range_and_qualifier_preserved():
    assert candidates('FEV1 30-40%')[0]['range_ambiguous']
    assert candidates('FEV1 <30%')[0]['qualifier'] == '<'


def test_no_date_or_distant_number_as_measurement():
    assert not candidates('PFT scheduled for 10/12/2026; FEV1 was not reported')
    assert not candidates('FEV1 improved after treatment on 10/12')
