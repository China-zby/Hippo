import os
from typing import Any
import yaml


class FlowList(list):
    pass


def flow_list_representer(dumper, data):
    return dumper.represent_sequence('tag:yaml.org,2002:seq', data, flow_style=True)


yaml.add_representer(FlowList, flow_list_representer)


class Config(object):
    def __init__(self, config_path, cache_path,
                 deviceid, dataname, datatype, objects,
                 methodname, scenename):
        self.config_path = config_path
        self.cache_path = cache_path
        self._store_in_cache(deviceid, dataname,
                             datatype, methodname,
                             objects, scenename)
        self.deviceid, self.dataname, \
            self.datatype, self.methodname, \
            self.objects, self.scenename = deviceid, dataname, \
            datatype, methodname, \
            objects, scenename

        self.data = None

    def _store_in_cache(self, deviceid, dataname, datatype, methodname, objects, scenename):
        # read config file
        with open(self.config_path, 'r') as file:
            data = yaml.safe_load(file)
        data['deviceid'] = deviceid
        data['methodname'] = methodname
        data['logbase']['scene'] = scenename
        data['detectbase']['classes'] = list(
            map(lambda x: x.lower(), objects))
        data['database']['dataname'] = dataname
        data['database']['datatype'] = datatype
        # update the list of config with flowlist
        for key in data.keys():
            if type(data[key]) == list:
                data[key] = FlowList(data[key])
            if type(data[key]) == dict:
                for subkey in data[key]:
                    if type(data[key][subkey]) == list:
                        data[key][subkey] = FlowList(data[key][subkey])
        # store in cache
        with open(self.cache_path, 'w') as file:
            yaml.dump(data, file)

    def update_cache(self, config):
        config['deviceid'] = self.deviceid
        config['methodname'] = self.methodname
        config['logbase']['scene'] = self.scenename
        config['detectbase']['classes'] = list(
            map(lambda x: x.lower(), self.objects))
        config['database']['dataname'] = self.dataname
        config['database']['datatype'] = self.datatype
        with open(self.cache_path, 'w') as file:
            yaml.dump(config, file)

    def load_cache(self):
        with open(self.cache_path, 'r') as file:
            data = yaml.safe_load(file)
        self.data = data
        return self.data.copy()

    def save_cache(self):
        with open(self.cache_path, 'w') as file:
            yaml.dump(self.load_cache(), file)

    @property
    def cache_dir(self):
        return os.path.dirname(self.cache_path)

    def save(self, path):
        with open(path, 'w') as file:
            yaml.dump(self.load_cache(), file)

    def __repr__(self):
        return self.cache_path

    def __call__(self):
        return self.load_cache()

def generate_random_config(config_path, cache_path, deviceid, dataname, datatype, objects, methodname, scenename):
    config = Config(config_path, cache_path, deviceid, dataname, datatype, objects, methodname, scenename)
    data = config.load_cache()
    data['detectbase']['classes'] = list(map(lambda x: x.lower(), objects))
    data['database']['dataname'] = dataname
    data['database']['datatype'] = datatype
    data['deviceid'] = deviceid
    data['methodname'] = methodname
    data['logbase']['scene'] = scenename
    config.update_cache(data)
    return config