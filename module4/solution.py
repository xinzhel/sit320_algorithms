class Node:
    def __init__(self, value):
        self.val = value
        self.right = None
        self.left = None
        self.height = 1
        self.balance = 0

    def updateBalance(self):
        if self.left and self.right:
            self.height = 1 + max(self.left.height, self.right.height)
            self.balance = self.left.height - self.right.height
        elif self.left:
            self.height = 1 + self.left.height
            self.balance = self.left.height
        elif self.right:
            self.height = 1 + self.right.height
            self.balance = 0 - self.right.height
        else:
            self.height = 1
            self.balance = 0


def checkBalance(nodes):
    if abs(nodes.balance) > 1:
        return False
    elif nodes.left and nodes.right:
        if abs(nodes.left.balance) > 1 or abs(nodes.right.balance) > 1:
            return False
        return checkBalance(nodes.left) and checkBalance(nodes.right)
    elif nodes.left:
        if abs(nodes.left.balance) > 1:
            return False
        return checkBalance(nodes.left)
    elif nodes.right:
        if abs(nodes.right.balance) > 1:
            return False
        return checkBalance(nodes.right)
    return True


def findParent(nodes, node):
    if node.val == nodes.val:
        # Node is the root, therefore no parent exists
        return None
    elif nodes.left.val == node.val or nodes.right.val == node.val:
        # Current root is the parent
        return nodes
    elif node.val < nodes.val:
        return findParent(nodes.left, node)
    else:
        return findParent(nodes.right, node)


def leftRotation(nodes, nodeP):
    # parent = findParent(nodes, nodeP)
    nodeQ = nodeP.right
    temp = nodeQ.left
    # Perform rotation
    nodeQ.left = nodeP
    nodeP.right = temp
    nodeP.updateBalance()
    nodeQ.updateBalance()
    # if parent:
    #     if nodeP.val < parent.val:
    #         parent.left = nodeQ
    #     else:
    #         parent.right = nodeQ
    #     while parent:
    #         parent.updateBalance()
    #         parent = findParent(nodes, parent)
    # else:
    #     nodes = nodeQ
    return nodeQ


def rightRotation(nodes, nodeP):
    # parent = findParent(nodes, nodeP)
    nodeQ = nodeP.left
    temp = nodeQ.right
    # Perform rotation
    nodeQ.right = nodeP
    nodeP.left = temp
    nodeP.updateBalance()
    nodeQ.updateBalance()
    # if parent:
    #     if nodeP.val < parent.val:
    #         parent.left = nodeQ
    #     else:
    #         parent.right = nodeQ
    #     while parent:
    #         parent.updateBalance()
    #         parent = findParent(nodes, parent)
    return nodeQ


def leftRightRotation(nodes, nodeP):
    leftRotation(nodes, nodeP.left)
    return rightRotation(nodes, nodeP)


def rightLeftRotation(nodes, nodeP):
    rightRotation(nodes, nodeP.right)
    return leftRotation(nodes, nodeP)








