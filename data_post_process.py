from openmdao.lib.casehandlers.api import CaseDataset, caseset_query_to_html

cds = CaseDataset('airline_allocation.json', 'json')

caseset_query_to_html(cds.data,'airline_allocation.html')

