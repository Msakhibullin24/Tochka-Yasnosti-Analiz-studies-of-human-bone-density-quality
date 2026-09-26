from dxaqc.dicom_io import DicomReadError
from external_validation import count_route


class StubAnalyzer:
    def __init__(self, outcome):
        self.outcome = outcome

    def analyze(self, image):
        if isinstance(self.outcome, Exception):
            raise self.outcome
        return {'region': self.outcome}


def test_external_router_counts_refusal_and_wrong_region_separately():
    counts = {'correct': 0, 'incorrect': 0, 'abstained': 0, 'refusal_codes': {}}
    count_route(StubAnalyzer('spine'), object(), 'spine', counts)
    count_route(StubAnalyzer('hip_left'), object(), 'spine', counts)
    count_route(StubAnalyzer(DicomReadError('UNCERTAIN_REGION', 'withheld')), object(), 'spine', counts)
    assert counts == {'correct': 1, 'incorrect': 1, 'abstained': 1,
                      'refusal_codes': {'UNCERTAIN_REGION': 1}}
