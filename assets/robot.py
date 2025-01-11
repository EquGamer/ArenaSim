from typing import Optional
import numpy as np
import math
import torch
import os
import omni

from omni.isaac.core.robots.robot import Robot
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.objects import FixedCuboid
from omni.isaac.core.utils.prims import get_prim_at_path
from omni.isaac.core.prims import XFormPrim
from pxr import UsdGeom, Gf, UsdShade, UsdPhysics


class GameBot(Robot):
    def __init__(
        self,
        prim_path: str,
        name: Optional[str] = "gamebot",
        translation: Optional[np.ndarray] = None,
        orientation: Optional[np.ndarray] = None,
    ) -> None:
        self._prim_path = prim_path

        self.root_path = os.path.abspath(os.path.join(os.path.abspath(__file__), "../.."))
        self._usd_path = os.path.join(self.root_path, "assets/game_bot/game_bot.usd")
        self._wheel_joint_idx = torch.arange(0, 4, dtype=torch.int32)
        add_reference_to_stage(self._usd_path, prim_path)
        self._stage = omni.usd.get_context().get_stage()

        super().__init__(
            prim_path=prim_path,
            name=name,
            translation=translation,
            orientation=orientation,
            articulation_controller=None,
        )

        if name[0:-1] == 'blue':
            ligth_path = f"{prim_path}/light/visuals"
            material_path = f"{prim_path}/Looks/material_blue"
            visual_material = UsdShade.Material(get_prim_at_path(material_path))
            binding_api = UsdShade.MaterialBindingAPI(get_prim_at_path(ligth_path))
            binding_api.Bind(visual_material, bindingStrength=UsdShade.Tokens.strongerThanDescendants)


class Field(XFormPrim):
    def __init__(
        self,
        prim_path: str,
        name: Optional[str] = "field",
    ) -> None:
        # x,y,length,width,height
        self.blocks = [
            [ 0.0,  2.4, 5.6, 0.2, 0.45],
            [ 0.0, -2.4, 5.6, 0.2, 0.45],
            [ 2.7,  0.0, 0.2, 4.6, 0.45],
            [-2.7,  0.0, 0.2, 4.6, 0.45],

            [ 1.2,  0.0, 0.8, 0.2, 0.4],
            [-1.2,  0.0, 0.8, 0.2, 0.4],
            [ 0.0,  1.2, 0.8, 0.2, 0.4],
            [ 0.0, -1.2, 0.8, 0.2, 0.4],

            [-2.3,  1.2, 0.6, 0.2, 0.4],
            [ 2.3, -1.2, 0.6, 0.2, 0.4],
            [-1.5, -2.0, 0.2, 0.6, 0.4],
            [ 1.5,  2.0, 0.2, 0.6, 0.4],
        ]
        super().__init__(
            prim_path=prim_path,
            name=name,
        )
        for i in range(len(self.blocks)):
            FixedCuboid(
                prim_path=f"{prim_path}/block{i}",
                name=f"block{i}",
                translation=np.array([self.blocks[i][0], self.blocks[i][1], self.blocks[i][4] / 2]),
                scale=np.array(self.blocks[i][2:5]),
            )
        return



