import tritonclient.http as httpclient

#ZeroTier connection ip to WILLIEPC
CONNECTION_IP = "192.168.191.129"

client = httpclient.InferenceServerClient(url=CONNECTION_IP+":8000")
print(CONNECTION_IP)
print(client.is_server_live())

