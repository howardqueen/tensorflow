def test():
    a = [1,2,3,4];
    b = [[1,2,3,4,5,6,7,8,9,0],[1,2,3,4,5,6,7,8,9,0],[1,2,3,4,5,6,7,8,9,0],[1,2,3,4,5,6,7,8,9,0]];
    c = [[[1,2,3,4,5,6,7,8,9,0],[1,2,3,4,5,6,7,8,9,0],[1,2,3,4,5,6,7,8,9,0],[1,2,3,4,5,6,7,8,9,0]],[[1,2,3,4,5,6,7,8,9,0],[1,2,3,4,5,6,7,8,9,0],[1,2,3,4,5,6,7,8,9,0],[1,2,3,4,5,6,7,8,9,0]],[[1,2,3,4,5,6,7,8,9,0],[1,2,3,4,5,6,7,8,9,0],[1,2,3,4,5,6,7,8,9,0],[1,2,3,4,5,6,7,8,9,0]]]
    print('---a---');
    printMartrix(a);
    print();
    print('---b---');
    printMartrix(b);
    print();
    print('---c---');
    printMartrix(c,'',5,lambda i: 1 if i > 0 else 0);
    
def printMartrix(martrix, itemSpliter = None, lineItems = None, itemReader = None):
    prettyMartrix(lambda s: print(s, end=''), martrix, itemSpliter, lineItems, itemReader);
    print();

def strMartrix(martrix, itemSpliter = None, lineItems = None, itemReader = None):
    strs = [];
    prettyMartrix(lambda s:strs.append(str(s)), martrix, itemSpliter, lineItems, itemReader);
    return ''.join(strs);

def prettyMartrix(write, martrix, itemSpliter = None, lineItems = None, itemReader = None, shapes = None, parentPrefix = None):
    if write == None:
        write = lambda s: print(s, end='');
    #if martrix == None:
    #    write("<class 'None'>");
    #    return;
    if len(martrix) == 0:
        write("[]");
        return;
    if itemSpliter == None:
        itemSpliter = ',';
    # -1: print thumbnail
    # 0: print all
    # 1: 1 item per line, with enter
    if lineItems == None:
        lineItems = -1;
    if itemReader == None:
        itemReader = lambda i:i;
    if shapes == None:
        shapes = lenMatrix(martrix);
    if shapes <= 0:
        write(str(type(martrix)));
        return;
    prefix = None;
    if parentPrefix == None:
        prefix = '';
    else:
        prefix = parentPrefix + ' ';
    write(prefix);
    write('[');
    length = len(martrix);
    if shapes == 1:
        # Print All
        if lineItems == 0 or (lineItems < 0 and length <= 3) or lineItems >= length:
            for i in range(length):
                write(itemReader(martrix[i]));
                write(itemSpliter);
            write(']');
            return;
        # Print Thumbnail
        if lineItems < 0:
            write(itemReader(martrix[0]));
            write(itemSpliter);
            write(' ... ');
            write(itemReader(martrix[length - 1]));
            write('] #');
            write(length);
            return;
        # Auto Wrap Lines
        index = 0;
        while index < length:
            write('\r\n ');
            write(prefix);
            for i in range(lineItems):
                if index < length:
                    write(itemReader(martrix[index]));
                    write(itemSpliter);
                index += 1;
        write('\r\n');
        write(prefix);
        write(']');
        return;
    # Print All
    if length <= 3: # lineItems == 0 or
        for i in range(length):
            write('\r\n');
            prettyMartrix(write, martrix[i], itemSpliter, lineItems, itemReader, shapes - 1, prefix);
            write(',');
        write('\r\n');
        write(prefix);
        write(']');
        return;
    # Print Thumbnail
    write('\r\n');
    write(prefix);
    write(' #1\r\n');
    prettyMartrix(write, martrix[0], itemSpliter, lineItems, itemReader, shapes - 1, prefix);
    write(',');
    write('\r\n');
    write(prefix);
    write(' ...\r\n');
    write(prefix);
    write(' #');
    write(length);
    write('\r\n');
    prettyMartrix(write, martrix[length - 1], itemSpliter, lineItems, itemReader, shapes - 1, prefix);
    write('\r\n');
    write(prefix);
    write(']');
    return;

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