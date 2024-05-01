from id3 import *

def check_simple_tree_1_depth():
    """Checks predictions and tree structure for 1-featured data, which can be learnt with 100% accuracy"""
    data = [[0], [1]]
    labels = ["A", "B"]

    tree = DecisionTree()
    tree.fit_id3(data, labels)

    assert tree.predict([0]) == "A"
    assert tree.predict([1]) == "B"

    assert tree.root_node.feature_idx == 0
    assert tree.root_node.label == None

    # tree structure checks
    assert len(tree.root_node.branches) == 2
    for _, node in tree.root_node.branches.items():
        assert node.feature_idx == -1
        assert node.label != None
        assert len(node.branches) == 0


def check_tree_deep_same_entropy():
    """Checks predictions and tree structure for 2-featured data, which can be learnt with 100% accuracy; both attributes have equal entropies"""
    data = [["A", "C"], ["A", "D"], ["B", "C"], ["B", "D"]]
    labels = [True, False, False, True]

    tree = DecisionTree()
    tree.fit_id3(data, labels)

    # check if predictions match the training data (they should for this data)
    predictions = [tree.predict(x) for x in data]
    assert predictions == labels


def check_tree_deep_variable_entropy():
    """Checks predictions and tree structure for 2-featured data, which can be learnt with 100% accuracy; first attribute has less entropy than the second one"""
    data = [["A", "C"], ["A", "D"], ["B", "C"], ["B", "D"]]
    labels = [True, False, False, False]

    tree = DecisionTree()
    tree.fit_id3(data, labels)

    # check if predictions match the training data (they should for this data)
    predictions = [tree.predict(x) for x in data]
    assert predictions == labels

    # first choice is made on the attribute with lesser entropy
    assert tree.root_node.feature_idx == 0

    branches = tree.root_node.branches
    
    # A branch has 2 subbranches
    node_A = branches["A"]
    len(node_A.branches) == 2
    node_A.label == None
    assert node_A.is_leaf() == False

    # B branch is a leaf node
    node_B = branches["B"]
    len(node_B.branches) == 0
    node_B.label == False
    assert node_B.is_leaf() == True


def check_tree_playgolf():
    """
    Checks the tree predictions and structure comparing it to a known valid example available at:
    https://github.com/milaan9/Python_Decision_Tree_and_Random_Forest/blob/main/001_Decision_Tree_PlayGolf_ID3.ipynb
    """

    # feature names to indexes
    OUTLOOK = 0
    TEMPERATURE = 1 
    HUMIDITY = 2
    WINDY = 3

    # Outlook Temperature Humidity Windy
    data = [
        ["Sunny", "Hot", "High", "Weak"],
        ["Sunny", "Hot", "High", "Strong"],
        ["Overcast", "Hot", "High", "Weak"],
        ["Rainy", "Mild", "High", "Weak"],
        ["Rainy", "Cool", "Normal", "Weak"],
        ["Rainy", "Cool", "Normal", "Strong"],
        ["Overcast", "Cool", "Normal", "Strong"],
        ["Sunny", "Mild", "High", "Weak"],
        ["Sunny", "Cool", "Normal", "Weak"],
        ["Rainy", "Mild", "Normal", "Weak"],
        ["Sunny", "Mild", "Normal", "Strong"],
        ["Overcast", "Mild", "High", "Strong"],
        ["Overcast", "Hot", "Normal", "Weak"],
        ["Rainy", "Mild", "High", "Strong"],
    ]

    labels = ["No", "No", "Yes", "Yes", "Yes", "No", "Yes", "No", "Yes", "Yes", "Yes", "Yes", "Yes", "No"]

    tree = DecisionTree()
    tree.fit_id3(data, labels)
    
    assert tree.root_node.feature_idx == OUTLOOK

    # Root node branches
    assert tree.root_node.branches["Sunny"].is_leaf() == False
    assert tree.root_node.branches["Overcast"].is_leaf() == True
    assert tree.root_node.branches["Rainy"].is_leaf() == False

    # Sunny node
    sunny_node = tree.root_node.branches["Sunny"]
    assert sunny_node.feature_idx == HUMIDITY

    assert sunny_node.branches["High"].is_leaf() == True
    assert sunny_node.branches["High"].label == "No"

    assert sunny_node.branches["Normal"].is_leaf() == True # Normal -> Low in reference material
    assert sunny_node.branches["Normal"].label == "Yes"

    # Overcast node
    overcast_node = tree.root_node.branches["Overcast"]
    assert overcast_node.label == "Yes"

    # Rain node
    rainy_node = tree.root_node.branches["Rainy"]
    assert rainy_node.feature_idx == WINDY
    assert rainy_node.branches["Strong"].is_leaf() == True
    assert rainy_node.branches["Strong"].label == "No"
    
    assert rainy_node.branches["Weak"].is_leaf() == True
    assert rainy_node.branches["Weak"].label == "Yes"
    

check_simple_tree_1_depth()
check_tree_deep_same_entropy()
check_tree_deep_variable_entropy()
check_tree_playgolf()