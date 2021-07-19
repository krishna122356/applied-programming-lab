# First Assignment, roll no - EE19B147
import sys
import numpy as np
import math

if (len(sys.argv) != 2):  # Checks number of args
    print("Ensure a single filename is entered")
    exit()
arg0 = sys.argv[1]
if (arg0[-8:] != ".netlist"):  # Checks extension
    print("Incorrect file type")
    exit()
try:  # Checks if the file exists
    f = open(arg0)
    lines = f.readlines()
except:
    print("Unable to locate file")
    exit()
noOfLines = len(lines)
circuitDetect = False
endDetect = False
CollectionOfLines = []
lineCount = 0
ACDetect = False
# Extracts the segment from .circuit to .end. If either does not exist, provides error as such.
for i in range(noOfLines):
    if (circuitDetect == False):
        lineCount += 1
        if (lines[i][:8] == ".circuit"):
            circuitDetect = True
    else:
        if (lines[i][:4] != '.end') and (endDetect == False):
            CollectionOfLines.append(lines[i])
        elif (endDetect == False):
            endDetect = True
        elif (lines[i][:3] == ".ac") and (ACDetect == False):
            ACDetect = True
            CollectionOfLines.append(lines[i])

if (circuitDetect == False) or (endDetect == False):
    print("Invalid Syntax,.circuit and/or .end not detected")
    exit()
NetlistCode = []  # Will contain correct code.
for eachLine in CollectionOfLines:
    lineCount += 1
    CurrentLine = eachLine.split()
    Tokens = []
    for token in CurrentLine:
        if (token[0] == "#"):  # Detects comments, and if present removes them.
            break
        else:
            Tokens.append(token)
    if (Tokens == []):  # Checks next line, in case an empty line is inserted in between code.
        continue
    # Check token type, and check for the possible errors : Number of parameters, alphanumeric node names, etc. If
    # Syntactically correct, reverses the List of tokens and stores them in string ThisLine, which is then stored in
    # in the list printout.
    Element = Tokens[0]
    if (Element[0] in ["R", "L", "C"]):  #
        if (len(Tokens) != 4):
            print("Error at line", lineCount)
            print(
                "Incorrect number of parameters for the specified element. There should only be 3 additional parametes.")
            exit()
        if (Tokens[1].isalnum() == False) or (Tokens[2].isalnum() == False):
            print("Error at line", lineCount)
            print("Node names should be AlphaNumeric.")
            exit()
        Tokens[3] = float(Tokens[3])
        NetlistCode.append(Tokens)
    elif (Element[0] == "V"):
        if (ACDetect == False):
            if (len(Tokens) != 4):
                print("Error at line", lineCount)
                print(
                    "Incorrect number of parameters for the specified element. There should only be 3 additional parametes.")
                exit()
            if (Tokens[1].isalnum() == False) or (Tokens[2].isalnum() == False):
                print("Error at line", lineCount)
                print("Node names should be AlphaNumeric.")
                exit()
            Tokens[3] = float(Tokens[3])
            NetlistCode.append(Tokens)
        else:
            if (len(Tokens) == 5) and (Tokens[3] == "dc"):
                if (Tokens[1].isalnum() == True) and (Tokens[2].isalnum() == True):
                    Tokens[4] = float(Tokens[4])
                    NetlistCode.append(Tokens)
                else:
                    print("Error at line", lineCount)
                    exit()
            elif (len(Tokens) == 6) and (Tokens[3] == "ac"):
                if (Tokens[1].isalnum() == True) and (Tokens[2].isalnum() == True):
                    Tokens[4] = float(Tokens[4])
                    Tokens[-1] = float(Tokens[-1])
                    NetlistCode.append(Tokens)
                else:
                    print("Error at line", lineCount)
                    exit()
    elif (Element[0] == "I"):
        if (ACDetect == False):
            if (len(Tokens) != 4):
                print("Error at line", lineCount)
                print(
                    "Incorrect number of parameters for the specified element. There should only be 3 additional parametes.")
                exit()
            if (Tokens[1].isalnum() == False) or (Tokens[2].isalnum() == False):
                print("Error at line", lineCount)
                print("Node names should be AlphaNumeric.")
                exit()
            Tokens[3] = float(Tokens[3])
            NetlistCode.append(Tokens)
        else:
            if (len(Tokens) == 5) and (Tokens[3] == "dc"):
                if (Tokens[1].isalnum() == True) and (Tokens[2].isalnum() == True):
                    Tokens[4] = float(Tokens[4])
                    NetlistCode.append(Tokens)
                else:
                    print("Error at line", lineCount)
                    exit()
            elif (len(Tokens) == 6) and (Tokens[3] == "ac"):
                if (Tokens[1].isalnum() == True) and (Tokens[2].isalnum() == True):
                    Tokens[4] = float(Tokens[4])
                    Tokens[-1] = float(Tokens[-1])
                    NetlistCode.append(Tokens)
                else:
                    print("Error at line", lineCount)
                    exit()

    elif (Element[0] in ["E", "G"]):
        if (len(Tokens) != 6):
            print("Error at line", lineCount)
            print(
                "Incorrect number of parameters for the specified element. There should only be 6 additional parametes.")
            exit()
        if (Tokens[1].isalnum() == False) or (Tokens[2].isalnum() == False) or (Tokens[3].isalnum() == False) or (
                Tokens[4].isalnum() == False):
            print("Error at line", lineCount)
            print("Node names should be AlphaNumeric.")
            exit()
        Tokens[3] = float(Tokens[5])
        NetlistCode.append(Tokens)
    elif (Element[0] in ["H", "F"]):
        if (len(Tokens) != 5):
            print("Error at line", lineCount)
            print(
                "Incorrect number of parameters for the specified element. There should only be 5 additional parametes.")
            exit()
        if (Tokens[3][0] != "V"):
            print("Error at line", lineCount)
            print("Voltage name is not present, Syntax is incorrect (???)")
            exit()
        if (Tokens[1].isalnum() == False) or (Tokens[2].isalnum() == False):
            print("Error at line", lineCount)
            print("Node names should be AlphaNumeric.")
            exit()
        Tokens[3] = float(Tokens[4])
        NetlistCode.append(Tokens)
    elif (Element == ".ac"):
        if (len(Tokens) != 3):
            print("Error at line", lineCount)
            print("incorrect .ac format")
            exit()
        frequency = float(Tokens[-1])
    else:
        print("Error at line", lineCount)
        print("No such component.")
        exit()
# Reverse printout.
f.close()  # Close the file.


# Detect and store for DC case.
def detectAndStore(Tokens, AllNodes, NewCurrents, NodeIndex, CurrentIndex, NodeCounter, CurrentCounter):
    Element = Tokens[0]
    if (Element[0] == "R"):
        n1 = Tokens[1]
        n2 = Tokens[2]
        if (n1 in AllNodes):
            AllNodes[n1].append([n2, Tokens[3], "Resistor"])
        else:
            NodeIndex[n1] = NodeCounter
            NodeCounter += 1
            AllNodes[n1] = [[n2, Tokens[3], "Resistor"]]
        if (n2 in AllNodes):
            AllNodes[n2].append([n1, Tokens[3], "Resistor"])
        else:
            NodeIndex[n2] = NodeCounter
            NodeCounter += 1
            AllNodes[n2] = [[n1, Tokens[3], "Resistor"]]
    elif(Element[0]=="L"):
        n1 = Tokens[1]
        n2 = Tokens[2]
        if (n1 in AllNodes):
            AllNodes[n1].append([n2, Tokens[3], "Inductor"])
        else:
            NodeIndex[n1] = NodeCounter
            NodeCounter += 1
            AllNodes[n1] = [[n2, Tokens[3], "Inductor"]]
        if (n2 in AllNodes):
            AllNodes[n2].append([n1, Tokens[3], "Inductor"])
        else:
            NodeIndex[n2] = NodeCounter
            NodeCounter += 1
            AllNodes[n2] = [[n1, Tokens[3], "Inductor"]]

    elif (Element[0] == "V"):
        n1 = Tokens[1]
        n2 = Tokens[2]
        NewCurrentName = "I" + Tokens[0]
        NewCurrents[NewCurrentName] = [n1, n2]
        if (n1 in AllNodes):
            AllNodes[n1].append([n2, Tokens[3], "Voltage", "I" + Tokens[0]])
        else:
            AllNodes[n1] = [[n2, Tokens[3], "Voltage", "I" + Tokens[0]]]
            NodeIndex[n1] = NodeCounter
            NodeCounter += 1
        if (n2 in AllNodes):
            AllNodes[n2].append([n1, Tokens[3], "Voltage", "I" + Tokens[0]])
        else:
            AllNodes[n2] = [[n1, Tokens[3], "Voltage", "I" + Tokens[0]]]
            NodeIndex[n2] = NodeCounter
            NodeCounter += 1
        NewCurrentName = "I" + Tokens[0]
        NewCurrents[NewCurrentName] = [n1, n2]
        CurrentIndex[NewCurrentName] = CurrentCounter
        CurrentCounter += 1
    elif (Element[0] == "I"):
        n1 = Tokens[1]
        n2 = Tokens[2]
        if (n1 in AllNodes):
            AllNodes[n1].append([n2, -Tokens[3], "Current"])
        else:
            NodeIndex[n1] = NodeCounter
            NodeCounter += 1
            AllNodes[n1] = [[n2, -Tokens[3], "Current"]]
        if (n2 in AllNodes):
            AllNodes[n2].append([n1, Tokens[3], "Current"])
        else:
            NodeIndex[n2] = NodeCounter
            NodeCounter += 1
            AllNodes[n2] = [[n1, Tokens[3], "Current"]]

    return CurrentCounter, NodeCounter

# Solver for DC case.
def DoDC(ListOfTokens):
    AllNodes = {}
    NewCurrents = {}
    NodeIndex = {}
    CurrentIndex = {}
    NodeCounter = 0
    CurrentCounter = 0

    # Traverses and stores data using detectAndStore function.
    for eachLine in ListOfTokens:
        CurrentCounter, NodeCounter = detectAndStore(eachLine, AllNodes, NewCurrents, NodeIndex, CurrentIndex,
                                                     NodeCounter, CurrentCounter)
    for i in CurrentIndex:
        CurrentIndex[i] += len(NodeIndex)
    Dimensions = len(AllNodes) + len(NewCurrents)
    M = np.zeros([Dimensions, Dimensions], dtype="float")
    b = np.zeros(Dimensions, dtype="float")
    # Go through all connected nodes of eachNode
    for eachNode in AllNodes:
        currentNode = NodeIndex[eachNode]
        # The connected nodes of eachNode is eachChildNode
        for eachChildNode in AllNodes[eachNode]:
            currentChildNode = NodeIndex[eachChildNode[0]]
            value = eachChildNode[1]
            # In case we detect a passive element. Only M array is changed
            if (eachChildNode[2] == "Resistor"):
                M[currentNode][currentNode] += 1 / value
                M[currentNode][currentChildNode] -= 1 / value
            # Inductor will be taken as negligible resistance.
            elif(eachChildNode[2]=="Inductor"):
                M[currentNode][currentNode]+=1/1e-20
                M[currentNode][currentChildNode]-=1/1e-20
            # In case we detect a Voltage source.
            elif (eachChildNode[2] == "Voltage"):
                if (NewCurrents[eachChildNode[3]][0] == eachNode):
                    M[currentNode][CurrentIndex[eachChildNode[3]]] += 1
                else:
                    M[currentNode][CurrentIndex[eachChildNode[3]]] -= 1
                # Here, we change by half instead of 1, as we are performing this step twice when we populate the M array.
                M[CurrentIndex[eachChildNode[3]]][NodeIndex[NewCurrents[eachChildNode[3]][0]]] -= 1 / 2
                M[CurrentIndex[eachChildNode[3]]][NodeIndex[NewCurrents[eachChildNode[3]][1]]] += 1 / 2
                b[CurrentIndex[eachChildNode[3]]] = value
            # In case we detect a Current source. Only b array is modified.
            elif (eachChildNode[2] == "Current"):
                b[currentNode] += value

    oneRow = [0 for i in range(len(M))]
    oneRow[NodeIndex["GND"]] = 1
    M[len(AllNodes) - 1] = oneRow
    b[len(AllNodes) - 1] = 0
    print("This is M matrix:")
    print(M)
    print("This is b matrix")
    print(b)

    reverseVoltDict = {}
    for i in NodeIndex:
        reverseVoltDict[NodeIndex[i]] = i
    reverseCurrDict = {}
    for i in CurrentIndex:
        reverseCurrDict[CurrentIndex[i]] = i
    try:
        x = np.linalg.solve(M, b)
        print(x)
        print("These are the results: ")
        for i in range(len(NodeIndex)):
            print("Voltage at Node", reverseVoltDict[i], "is: -", x[i])
        for i in range(len(NodeIndex), len(NodeIndex) + len(CurrentIndex)):
            print("Current via voltage source", reverseCurrDict[i][1:], "is: -", x[i])

    except:
        print("Unsolvable matrix. Some params are off.")
        exit()

#Detect and Store for AC case.
def detectAndStoreAC(Tokens, AllNodes, NewCurrents, NodeIndex, CurrentIndex, NodeCounter, CurrentCounter, frequency):
    Element = Tokens[0]
    if (Element[0] == "R"):
        n1 = Tokens[1]
        n2 = Tokens[2]
        if (n1 in AllNodes):
            AllNodes[n1].append([n2, Tokens[3], "Resistor"])
        else:
            NodeIndex[n1] = NodeCounter
            NodeCounter += 1
            AllNodes[n1] = [[n2, Tokens[3], "Resistor"]]
        if (n2 in AllNodes):
            AllNodes[n2].append([n1, Tokens[3], "Resistor"])
        else:
            NodeIndex[n2] = NodeCounter
            NodeCounter += 1
            AllNodes[n2] = [[n1, Tokens[3], "Resistor"]]
    elif (Element[0] == "C"):
        n1 = Tokens[1]
        n2 = Tokens[2]
        if (n1 in AllNodes):
            AllNodes[n1].append([n2, complex(0, -1 / (Tokens[3] * frequency)), "Capacitor"])
        else:
            NodeIndex[n1] = NodeCounter
            NodeCounter += 1
            AllNodes[n1] = [[n2, complex(0, -1 / (Tokens[3] * frequency)), "Capacitor"]]
        if (n2 in AllNodes):
            AllNodes[n2].append([n1, complex(0, -1 / (Tokens[3] * frequency)), "Capacitor"])
        else:
            NodeIndex[n2] = NodeCounter
            NodeCounter += 1
            AllNodes[n2] = [[n1, complex(0, -1 / (Tokens[3] * frequency)), "Capacitor"]]
    elif (Element[0] == "L"):
        n1 = Tokens[1]
        n2 = Tokens[2]
        if (n1 in AllNodes):
            AllNodes[n1].append([n2, complex(0, Tokens[3] * frequency), "Inductor"])
        else:
            NodeIndex[n1] = NodeCounter
            NodeCounter += 1
            AllNodes[n1] = [[n2, complex(0, Tokens[3] * frequency), "Inductor"]]
        if (n2 in AllNodes):
            AllNodes[n2].append([n1, complex(0, Tokens[3] * frequency), "Inductor"])
        else:
            NodeIndex[n2] = NodeCounter
            NodeCounter += 1
            AllNodes[n2] = [[n1, complex(0, Tokens[3] * frequency), "Inductor"]]

    elif (Element[0] == "V"):
        n1 = Tokens[1]
        n2 = Tokens[2]
        if (Tokens[3] == "dc"):
            NewCurrentName = "I" + Tokens[0]
            NewCurrents[NewCurrentName] = [n1, n2]
            if (n1 in AllNodes):
                AllNodes[n1].append([n2, Tokens[4], "Voltage", "I" + Tokens[0]])
            else:
                AllNodes[n1] = [[n2, Tokens[4], "Voltage", "I" + Tokens[0]]]
                NodeIndex[n1] = NodeCounter
                NodeCounter += 1
            if (n2 in AllNodes):
                AllNodes[n2].append([n1, Tokens[4], "Voltage", "I" + Tokens[0]])
            else:
                AllNodes[n2] = [[n1, Tokens[4], "Voltage", "I" + Tokens[0]]]
                NodeIndex[n2] = NodeCounter
                NodeCounter += 1
            NewCurrentName = "I" + Tokens[0]
            NewCurrents[NewCurrentName] = [n1, n2]
            CurrentIndex[NewCurrentName] = CurrentCounter
            CurrentCounter += 1
        else:
            NewCurrentName = "I" + Tokens[0]
            NewCurrents[NewCurrentName] = [n1, n2]
            phase = Tokens[-1]
            Tokens[4] /= 2
            if (n1 in AllNodes):
                AllNodes[n1].append(
                    [n2, complex(Tokens[4] * math.cos(phase), Tokens[4] * math.sin(phase)), "Voltage", "I" + Tokens[0]])
            else:
                AllNodes[n1] = [
                    [n2, complex(Tokens[4] * math.cos(phase), Tokens[4] * math.sin(phase)), "Voltage", "I" + Tokens[0]]]
                NodeIndex[n1] = NodeCounter
                NodeCounter += 1
            if (n2 in AllNodes):
                AllNodes[n2].append(
                    [n1, complex(Tokens[4] * math.cos(phase), Tokens[4] * math.sin(phase)), "Voltage", "I" + Tokens[0]])
            else:
                AllNodes[n2] = [
                    [n1, complex(Tokens[4] * math.cos(phase), Tokens[4] * math.sin(phase)), "Voltage", "I" + Tokens[0]]]
                NodeIndex[n2] = NodeCounter
                NodeCounter += 1
            NewCurrentName = "I" + Tokens[0]
            NewCurrents[NewCurrentName] = [n1, n2]
            CurrentIndex[NewCurrentName] = CurrentCounter
            CurrentCounter += 1
    elif (Element[0] == "I"):
        n1 = Tokens[1]
        n2 = Tokens[2]
        if (Tokens[3] == "dc"):
            if (n1 in AllNodes):
                AllNodes[n1].append([n2, -Tokens[4], "Current"])
            else:
                NodeIndex[n1] = NodeCounter
                NodeCounter += 1
                AllNodes[n1] = [[n2, -Tokens[4], "Current"]]
            if (n2 in AllNodes):
                AllNodes[n2].append([n1, Tokens[4], "Current"])
            else:
                NodeIndex[n2] = NodeCounter
                NodeCounter += 1
                AllNodes[n2] = [[n1, Tokens[4], "Current"]]
        elif (Tokens[3] == "ac"):
            Tokens[4] /= 2
            phase = Tokens[-1]
            if (n1 in AllNodes):
                AllNodes[n1].append([n2, -complex(Tokens[4] * math.cos(phase), Tokens[4] * math.sin(phase)), "Current"])
            else:
                NodeIndex[n1] = NodeCounter
                NodeCounter += 1
                AllNodes[n1] = [[n2, -complex(Tokens[4] * math.cos(phase), Tokens[4] * math.sin(phase)), "Current"]]
            if (n2 in AllNodes):
                AllNodes[n2].append([n1, complex(Tokens[4] * math.cos(phase), Tokens[4] * math.sin(phase)), "Current"])
            else:
                NodeIndex[n2] = NodeCounter
                NodeCounter += 1
                AllNodes[n2] = [[n1, complex(Tokens[4] * math.cos(phase), Tokens[4] * math.sin(phase)), "Current"]]

    return CurrentCounter, NodeCounter

# Solve for AC case.
def DoAC(ListOfTokens, frequency):
    AllNodes = {}
    NewCurrents = {}
    NodeIndex = {}
    CurrentIndex = {}
    NodeCounter = 0
    CurrentCounter = 0

    # Traverses and stores data using detectAndStoreAC function.
    for eachLine in ListOfTokens:
        CurrentCounter, NodeCounter = detectAndStoreAC(eachLine, AllNodes, NewCurrents, NodeIndex, CurrentIndex,
                                                       NodeCounter, CurrentCounter, 2 * np.pi * frequency)
    for i in CurrentIndex:
        CurrentIndex[i] += len(NodeIndex)
    Dimensions = len(AllNodes) + len(NewCurrents)
    M = np.zeros([Dimensions, Dimensions], dtype="complex")
    b = np.zeros(Dimensions, dtype="complex")

    # Go through all connected nodes of eachNode
    for eachNode in AllNodes:
        currentNode = NodeIndex[eachNode]
        # The connected nodes of eachNode is eachChildNode
        for eachChildNode in AllNodes[eachNode]:
            currentChildNode = NodeIndex[eachChildNode[0]]
            value = eachChildNode[1]

            # In case we detect a passive element. Only M array is changed
            if (eachChildNode[2] == "Resistor") or (eachChildNode[2] == "Inductor") or (
                    eachChildNode[2] == "Capacitor"):
                M[currentNode][currentNode] += 1 / value
                M[currentNode][currentChildNode] -= 1 / value
            # For a voltage source
            elif (eachChildNode[2] == "Voltage"):
                if (NewCurrents[eachChildNode[3]][0] == eachNode):
                    M[currentNode][CurrentIndex[eachChildNode[3]]] += 1
                else:
                    M[currentNode][CurrentIndex[eachChildNode[3]]] -= 1
                #Here, we change by half instead of 1, as we are performing this step twice when we populate the M array.
                M[CurrentIndex[eachChildNode[3]]][NodeIndex[NewCurrents[eachChildNode[3]][0]]] -= 1 / 2
                M[CurrentIndex[eachChildNode[3]]][NodeIndex[NewCurrents[eachChildNode[3]][1]]] += 1 / 2
                b[CurrentIndex[eachChildNode[3]]] = value
            # For a current source. Only b array is changed
            elif (eachChildNode[2] == "Current"):
                b[currentNode] += value

    oneRow = [0 for i in range(len(M))]
    oneRow[NodeIndex["GND"]] = 1
    M[len(AllNodes) - 1] = oneRow
    b[len(AllNodes) - 1] = 0
    print("This is M matrix:")
    print(M)
    print("This is b matrix")
    print(b)

    reverseVoltDict = {}
    for i in NodeIndex:
        reverseVoltDict[NodeIndex[i]] = i
    reverseCurrDict = {}
    for i in CurrentIndex:
        reverseCurrDict[CurrentIndex[i]] = i
    try:

        x = np.linalg.solve(M, b)
        print(x)
        print("These are the results: ")
        for i in range(len(NodeIndex)):
            print("Voltage at Node", reverseVoltDict[i], "is: -", x[i])
        for i in range(len(NodeIndex), len(NodeIndex) + len(CurrentIndex)):
            print("Current via voltage source", reverseCurrDict[i][1:], "is: -", x[i])

    except:
        print("Unsolvable matrix. some params are off.")
        exit()

if (ACDetect == False):
    DoDC(NetlistCode)
else:
    DoAC(NetlistCode, frequency)

# Krishna Somasundaram, Roll no EE19B147
