import argilla as rg
def auth():
    rg.init(
        api_url="http://procyon.tce.pi.gov.br:6900",
        api_key="admin.apikey",
        workspace='admin'
    )
    return rg