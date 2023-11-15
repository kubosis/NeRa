from loguru import logger
import time
from typing import Callable
from sshtunnel import SSHTunnelForwarder

from PyBasketballGNN.utils import check_input


def time_debug(func: Callable):
    """ Decorator that reports the execution time for debugging """

    def wrap(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()

        logger.debug(f"{func.__name__} execution took {end - start}")
        return result

    return wrap


def ssh_tunnel(fun):
    """ Decorator for creating and closing ssh tunnel, gives ssh_server to decorated function

    Mandatory keyword args:
       | ssh_host (str): host ssh server
       | ssh_user (str): ssh username
       | ssh_pkey (str): path to ssh private key

    Optional keyword args:
       | local_port (int): - local binding port 5432 by default
       | remote_port (int): - remote binding port 22 by default
       | local_address (str): - local bind address '127.0.0.1' by default

    important: include any *args, **kwargs that are used by decorated function!
    """

    def wrap(*args, **kwargs):
        ssh_host, ssh_user, ssh_pkey = check_input(['ssh_host', 'ssh_user', 'ssh_pkey'], **kwargs)

        local_port: int = kwargs.pop('local_port') if 'local_port' in kwargs else 5432
        remote_port: int = kwargs.pop('remote_port') if 'remote_port' in kwargs else 12345
        local_address: str = kwargs.pop('local_address') if 'local_address' in kwargs else '127.0.0.1'

        logger.info(f"Trying to establish connection with host {ssh_host} for user {ssh_user}")

        with SSHTunnelForwarder(
                (ssh_host, remote_port),
                ssh_username=ssh_user,
                ssh_pkey=ssh_pkey,
                remote_bind_address=(local_address, local_port)
        ) as server:
            server.start()  # start ssh sever
            logger.info(f"Server {ssh_host} connected via ssh")

            kwargs['ssh_server'] = server  # give reference of server to decorated fun
            result = fun(*args, **kwargs)

        return result

    return wrap
