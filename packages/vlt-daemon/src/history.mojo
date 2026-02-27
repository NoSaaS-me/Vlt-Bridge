struct CircularBuffer:
    var data: List[String]
    var size: Int
    var capacity: Int

    fn __init__(inout self, capacity: Int):
        self.data = List[String]()
        self.size = 0
        self.capacity = capacity

    fn push(inout self, item: String):
        if self.size >= self.capacity:
            _ = self.data.pop(0)
        else:
            self.size += 1
        self.data.append(item)

    fn get_all(self) -> String:
        var result = String("")
        for i in range(self.size):
            result += self.data[i]
        return result
