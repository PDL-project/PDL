import copy
from core import Controller
from Scenes.scene_initializer import BaseSceneInitializer
from misc import Arg

class SceneInitializer(BaseSceneInitializer):
    # 30 x 30, 6 fires (3x sand, 3x water) + 4 persons -> ~10 high-level subtasks
    # names: ReservoirSierra, ReservoirDelta, DepositFacility,
    #        RidgeFire, CanyonFire, MoorFire, HarborFire, PlainFire, GroveFire,
    #        LostPersonNora, LostPersonLeo, LostPersonMaya, LostPersonEthan

    def __init__(self) -> None:
        self.name='area_900_3a_3b_fires_4_persons'
        self.params={
                'grid_size' : (30,30,1),

                'reservoirs' : [
                    Arg(tp='a', name='ReservoirSierra', position=(10,25)),
                    Arg(tp='b', name='ReservoirDelta',  position=(7,22))
                    ],
                'deposits' : [
                    Arg(name='DepositFacility', position=(20,27)),
                    ],
                'fires' : [
                    Arg(tp='a', amt_light=1, amt_regions=None, enclosing_grid=(5,5), name='RidgeFire',  position=(3,3)),
                    Arg(tp='a', amt_light=1, amt_regions=None, enclosing_grid=(5,5), name='CanyonFire', position=(3,16)),
                    Arg(tp='a', amt_light=1, amt_regions=None, enclosing_grid=(5,5), name='MoorFire',   position=(15,3)),
                    Arg(tp='b', amt_light=1, amt_regions=None, enclosing_grid=(5,5), name='HarborFire', position=(15,16)),
                    Arg(tp='b', amt_light=1, amt_regions=None, enclosing_grid=(5,5), name='PlainFire',  position=(24,6)),
                    Arg(tp='b', amt_light=1, amt_regions=None, enclosing_grid=(5,5), name='GroveFire',  position=(24,20)),
                    ],
                'persons' : [
                    Arg(extra_load=0, find_probability=.3, name='LostPersonNora',  position=(5,28)),
                    Arg(extra_load=0, find_probability=.3, name='LostPersonLeo',   position=(27,26)),
                    Arg(extra_load=0, find_probability=.3, name='LostPersonMaya',  position=(28,14)),
                    Arg(extra_load=0, find_probability=.3, name='LostPersonEthan', position=(28,3)),
                    ],
                'agents' : [
                    Arg(position=(8,10)),
                    Arg(position=(10,16)),
                    Arg(position=(20,10)),
                    Arg(position=(5,20)),
                    Arg(position=(18,24)),
                    Arg(position=(15,27)),
                    Arg(position=(7,7)),
                    Arg(position=(3,27)),
                    Arg(position=(27,3)),
                    Arg(position=(25,28))
                    ]
                }
        self.task_timeout=60
        super().__init__()
