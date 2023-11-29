versions = ["1.0.1",'1.12.1',"1.6.1"]

sample_version = ['1','6','0']

#versions: list[str] = api.get_registry_model_versions(self.comet_workspace, model)
#formated_versions = map(partial(versions)
formated_versions = [ver.split('.') for ver in versions]
print(formated_versions)
# np_vers = np.array(formated_versions)
# np.unr
try:
    match_ds_ver = [x for x in formated_versions if x[1]==str(6)]
    print(match_ds_ver)
    max_2_vers = max(match_ds_ver, key= lambda x: x[2])
    print(isinstance(any(max_2_vers), list))
    if not isinstance(any(max_2_vers), list):
        sample_version[2] = str(int(max_2_vers[2])+1)
    else:
        sample_version[2] = str(max_2_vers[0][2])
    # sample_version[2] == str(max_2_vers[0][2]) if isinstance(any(max_2_vers), list) else str((int(max_2_vers[2])+1))
    print('.'.join(sample_version))
except:
    print( '.'.join(sample_version))

model = "yolov8l.pt"
print(model.removesuffix(".pt"))