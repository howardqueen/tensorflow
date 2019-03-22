def test():
    a = [1,2,3,4];
    b = [[1,2,3,4,5,6,7,8,9,0],[1,2,3,4,5,6,7,8,9,0],[1,2,3,4,5,6,7,8,9,0],[1,2,3,4,5,6,7,8,9,0]];
    print(strMatrix(a));
    print(strMatrix(b,'',5,lambda i: 1 if i > 0 else 0));

def strMatrix(martrix, 
    # None: print all
    # 0: print thumbnail
    # 10: 10 items per line, with enter
    spliteChar = ',',
    lineLength = 0,
    itemReader = lambda i:i,
    shapes = None):
    
    if shapes == None:
        shapes = lenMatrix(martrix);
    if shapes <= 0:
        return str(type(martrix));
    if len(martrix) == 0:
        return "[]";

    strs = [];
    length = len(martrix);
    if shapes == 1:
        # Print All
        if lineLength == None or (lineLength == 0 and length <= 3) or lineLength >= length:
            strs.append('[');
            for i in range(length):
                strs.append(str(itemReader(martrix[i])));
                strs.append(spliteChar);
            strs.append(']');
            return ''.join(strs);
        # Print Thumbnail
        elif lineLength == 0:
            strs.append('[');
            strs.append(str(itemReader(martrix[0])));
            strs.append(spliteChar);
            strs.append('...');
            strs.append(spliteChar);
            strs.append(str(itemReader(martrix[length - 1])));
            strs.append(']');
            return ''.join(strs);
        # Auto Wrap Lines
        strs.append('[\r\n');
        index = 0;
        while index < length:
            for i in range(lineLength):
                if index < length:
                    strs.append(str(itemReader(martrix[index])));
                    strs.append(spliteChar);
                index += 1;
            strs.append('\r\n');
        strs.append(']');
        return ''.join(strs);
    # Print All
    if lineLength == None or length <= 3:
        strs.append('[');
        for i in range(length):
            strs.append(strMatrix(martrix[i], spliteChar, lineLength, itemReader, shapes - 1));
            strs.append(spliteChar);
        strs.append(']');
        return ''.join(strs);
    # Print Thumbnail
    strs.append('[');
    strs.append(strMatrix(martrix[0], spliteChar, lineLength, itemReader, shapes - 1));
    strs.append(spliteChar);
    strs.append('...');
    strs.append(spliteChar);
    strs.append(strMatrix(martrix[length - 1], spliteChar, lineLength, itemReader, shapes - 1));
    strs.append(']');
    return ''.join(strs);

def lenMatrix(martrix):
    if isMatrix(martrix):
        if len(martrix) > 0:
            return 1 + lenMatrix(martrix[0]);
        return 1;
    return 0;

def isMatrix(martrix):
    t = str(type(martrix));
    if t == "<class 'list'>" or t == "<class 'numpy.ndarray'>":
        return True;
    return False;
    
#test();