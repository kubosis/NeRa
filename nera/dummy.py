from datetime import timedelta, datetime

import pandas as pd


class Dummy:
    id_list = [0, 1, 2]

    conf_len = {
        0: 3,
        1: 3,
        2: 3,
    }

    @staticmethod
    def _resolve_conf(conf: str) -> list:
        out = [[], [], []]
        resolved = {
            'h': ['home', 10, 0],
            'a': ['away', 0, 10],
            'd': ['draw', 5, 5]
        }
        for c in conf.lower():
            assert c in ['h', 'a', 'd']
            res = resolved[c]
            out[0].append(res[0])
            out[1].append(res[1])
            out[2].append(res[2])
        return out

    @staticmethod
    def _dummy0(conf: str = "hhh"):
        assert len(conf) == 3

        winner, home_pts, away_pts = Dummy._resolve_conf(conf)
        delta = timedelta(days=1)
        now = datetime.now()
        data = pd.DataFrame({'DT': [*[now + j * delta for j in range(3)]],
                             'Home': ['A', 'B', 'C'],
                             'Away': ['B', 'C', 'D'],
                             'Winner': winner,
                             'Home_points': home_pts,
                             'Away_points': away_pts,
                             'League': [*(3 * ['liga'])],
                             })
        return data

    @staticmethod
    def _dummy1(conf: str = "hhh"):
        assert len(conf) == 3

        winner, home_pts, away_pts = Dummy._resolve_conf(conf)
        delta = timedelta(days=1)
        now = datetime.now()
        data = pd.DataFrame({'DT': [*[now + j * delta for j in range(3)]],
                             'Home': ['A', 'B', 'C',],
                             'Away': ['B', 'C', 'A',],
                             'Winner': winner,
                             'Home_points': home_pts,
                             'Away_points': away_pts,
                             'League': [*(3 * ['liga'])],
                             })
        return data

    @staticmethod
    def _dummy2(conf: str = "hhh"):
        assert len(conf) == 3

        winner, home_pts, away_pts = Dummy._resolve_conf(conf)
        delta = timedelta(days=1)
        now = datetime.now()
        data = pd.DataFrame({'DT': [*[now + j * delta for j in range(3)]],
                             'Home': ['A', 'A', 'B',],
                             'Away': ['B', 'C', 'C',],
                             'Winner': winner,
                             'Home_points': home_pts,
                             'Away_points': away_pts,
                             'League': [*(3 * ['liga'])],
                             })
        return data

    @staticmethod
    def generate_dummy(match_count: int, team_count: int, conf: str):
        """ generate dummy dataset of arbitrary length and configuration (every match still within one league) """
        assert len(conf) == match_count

        teams = [chr(i+ord('A')) for i in range(team_count)]
        home = [teams[i % len(teams)] for i in range(match_count)]
        away = [teams[(i+1) % len(teams)] for i in range(match_count)]
        winner, home_pts, away_pts = Dummy._resolve_conf(conf)
        delta = timedelta(days=1)
        now = datetime.now()
        data = pd.DataFrame({'DT': [*[now + j * delta for j in range(match_count)]],
                             'Home': home,
                             'Away': away,
                             'Winner': winner,
                             'Home_points': home_pts,
                             'Away_points': away_pts,
                             'League': [*(match_count * ['liga'])],
                             })
        return data


def get_dummy_df(id_=0, conf="hhh"):
    assert id_ in Dummy.id_list
    return getattr(Dummy, f"_dummy{id_}")(conf)


def get_dummy_conf_len(id_: int):
    assert id_ in Dummy.id_list
    return Dummy.conf_len[id_]
