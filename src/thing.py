"""
Class for defining & initializing object meshes in Pybullet simulations.
"""

import pybullet as p


class Thing:
    def __init__(self, name, position, globalScaling):
        self.name = name
        self.initPos = position
        self.objectScale = globalScaling    # also defined in the arg85.yaml file, for shrinking the size of the mug
        self.initOrientation = p.getQuaternionFromEuler([0,0,0])
        self.urdf_path = "./src/urdf/objects/"

        if name == "block" or name == "block1":
            self.urdf_path += "blocks/block1/block1.urdf"
        elif name == "block2":
            self.urdf_path += "blocks/block2/block2.urdf"
        elif name == "block3":
            self.urdf_path += "blocks/block3/block3.urdf"
        elif name == "cylinder1":
            self.urdf_path += "cylinders/cylinder1/cylinder1.urdf"
        elif name == "cylinder2":
            self.urdf_path += "cylinders/cylinder2/cylinder2.urdf"
        elif name == "cylinder3":
            self.urdf_path += "cylinders/cylinder3/cylinder3.urdf"
        elif name == "bottle1":
            self.urdf_path += "bottles/bottle1/bottle1.urdf"
        elif name == "bottle2":
            self.urdf_path += "bottles/bottle2/bottle2.urdf"
        elif name == "bottle3":
            self.urdf_path += "bottles/bottle3/bottle3.urdf"

        id = p.loadURDF(self.urdf_path, self.initPos, self.initOrientation, useFixedBase=False, globalScaling=self.objectScale)
        self.setID(id)
        self.resetVelocity()

    def setInitPos(self,pos):
        self.initPos=pos

    def getInitPos(self):
        return self.initPos

    def setID(self,ID):
        self.ID=ID

    def getPos(self):
        bpao=p.getBasePositionAndOrientation(self.ID)
        return bpao[0]

    def setHeight(self,height):
        self.height=height

    def getHeight(self):
        return self.height

    def setWidth(self,width):
        self.width=width

    def getWidth(self):
        return self.width

    def setInitOrientation(self, quat):
        self.initOrientation=quat
    
    def resetVelocity(self):
        p.resetBaseVelocity(self.ID, [0, 0, 0], [0, 0, 0])
    
    def resetObject(self):
        pos = (0, 0, 0)
        orn = p.getQuaternionFromEuler([0,0,0])
        p.resetBasePositionAndOrientation(self.ID, pos, orn)
        self.resetVelocity()

    def getInitOrientation(self):
        return self.initOrientation 