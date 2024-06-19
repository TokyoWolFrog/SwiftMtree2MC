import Foundation
import Accelerate

public class MTreeNode {
    var entries: [MTreeEntry]
    var isLeaf: Bool
    
    var pEntry: MTreeEntry?
    
    init(isLeaf: Bool, pEntry: MTreeEntry? = nil) {
        self.entries = []
        self.isLeaf = isLeaf
        self.pEntry = pEntry
    }
    
    func toJSON() -> [String: Any] {
        var json = [String: Any]()
        json["isLeaf"] = String(self.isLeaf)
        var entriesJSON = [String: Any]()
        for (index, entry) in self.entries.enumerated() {
            entriesJSON[String(index)] = entry.toJSON()
        }
        json["entries"] = entriesJSON
        return json
    }
}

public class MTreeEntry {
    var mObject: [Float]
    var oid: String
    var coverRadius: Float
    var subTreeNode: MTreeNode?
    var pDistance: Float
    
    weak var mainNode: MTreeNode?
    
    init(mObject: [Float], oid: String, 
         subTreeNode: MTreeNode? = nil,
         coverRadius: Float = -0.1, pDistance: Float = -0.1,
         mainNode: MTreeNode? = nil) {
        self.mObject = mObject
        self.oid = oid
        self.coverRadius = coverRadius
        self.subTreeNode = subTreeNode
        self.pDistance = pDistance
        self.mainNode = mainNode
    }
    
    func toJSON() -> [String: Any] {
        var json = [String: Any]()
        json["oid"] = self.oid
        json["coverRadius"] = String(self.coverRadius)
        json["pDistance"] = String(self.pDistance)
        json["mObject"] = self.mObject
        if let subTreeNode = self.subTreeNode {
            json["subTreeNode"] = subTreeNode.toJSON()
        } else {
            json["subTreeNode"] = "None"
        }
        return json
    }
}

public class MTree {
    var root: MTreeNode
    var maxEntries: Int
    // kNNSearch
    var priorityQueue = [(node: MTreeNode, dmin: Float)]()
    var nearestNeighbors = [(oid: String, distance: Float)]()
    var dk = Float.infinity
    // The number of centroids
    let kmeans = 2
    let tolerance = 1
    let maximumIterations = 5
    var centroids = [[Float]]()
    var dataset = [[Float]]()
    var distances: UnsafeMutableBufferPointer<Float>!
    let centroidIndicesDescriptor: BNNSNDArrayDescriptor
    
    init(maxEntries: Int = 10) {
        self.root = MTreeNode(isLeaf: true)
        self.maxEntries = maxEntries
        self.distances = UnsafeMutableBufferPointer<Float>.allocate(capacity: (maxEntries + 1)*self.kmeans)
        centroidIndicesDescriptor = BNNSNDArrayDescriptor.allocateUninitialized(
            scalarType: Int32.self,
            shape: .matrixRowMajor(maxEntries + 1, 1))
    }
    
    deinit {
        distances.deallocate()
        centroidIndicesDescriptor.deallocate()
    }
        
    func insert(mObject: [Float], oid: String) {
        let entry = MTreeEntry(mObject: mObject, oid: oid)
        insert(entry: entry, into: self.root)
    }
    
    private func insert(entry: MTreeEntry, into node: MTreeNode) {
        if node.isLeaf {
            if node.entries.count == self.maxEntries {
                split(node: node, entry: entry)
            } else {
                storeEntryinNode(mainNode: node, newEntry: entry)
            }
        } else {
            let distancesAndMinusRadius = node.entries.compactMap { nodeEntry -> (Float, Float) in
                let distance = distanceFunc(nodeEntry.mObject, entry.mObject)
                return (distance, distance - nodeEntry.coverRadius)
            }
            
            if let (minIndex, minElement) = distancesAndMinusRadius.enumerated().filter({ $0.element.1 <= 0 }).map({ ($0.offset, $0.element.0) }).min(by: { $0.1 < $1.1 }) {
                entry.pDistance = minElement
                insert(entry: entry, into: node.entries[minIndex].subTreeNode!)
            } else {
                if let (minIndex, _) = distancesAndMinusRadius.enumerated().map({ ($0.offset, $0.element.1) }).min(by: { $0.1 < $1.1 }) {
                    node.entries[minIndex].coverRadius = distancesAndMinusRadius[minIndex].0
                    entry.pDistance = distancesAndMinusRadius[minIndex].0
                    insert(entry: entry, into: node.entries[minIndex].subTreeNode!)
                }
            }
        }
    }
    
    func distanceFunc(_ pointA: [Float], _ pointB: [Float]) -> Float {
        let distanceSquared = vDSP.distanceSquared(pointA, pointB)
        return sqrt(distanceSquared)
    }
    
    private func split(node: MTreeNode, entry: MTreeEntry) {
        storeEntryinNode(mainNode: node, newEntry: entry)
        let (parentObj1, parentEntry2) = promoteAndPartition(node: node)

        if let parentEntry = node.pEntry {
            parentEntry.mObject = parentObj1
            let maxParentDistance = node.isLeaf ? node.entries.map { $0.pDistance }.max() : node.entries.map { $0.pDistance + $0.coverRadius}.max()
            parentEntry.coverRadius = maxParentDistance!
            
            if let mainNode = parentEntry.mainNode {
                if let ppEntry = mainNode.pEntry {
                    parentEntry.pDistance = distanceFunc(parentEntry.mObject, ppEntry.mObject)
                }
                if mainNode.entries.count == self.maxEntries {
                    split(node: mainNode, entry: parentEntry2)
                } else {
                    storeEntryinNode(mainNode: mainNode, newEntry: parentEntry2)
                }
            }
        } else {
            let parentEntry1 = newPEntry(mObject: parentObj1, subTreeNode: node)
            
            let newRoot = MTreeNode(isLeaf: false)
            storeEntryinNode(mainNode: newRoot, newEntry: parentEntry1)
            storeEntryinNode(mainNode: newRoot, newEntry: parentEntry2)
            self.root = newRoot
        }
    }
    
    private func promoteAndPartition(node: MTreeNode) -> ([Float], MTreeEntry) {
        self.dataset.removeAll()
        
        for entry in node.entries {
            self.dataset.append(entry.mObject)
        }
        
        initializeCentroids()
        update()
        
        let entriesCount = node.entries.count
        var indicesToPop = IndexSet()
        let centroidIndices = self.centroidIndicesDescriptor.makeArray(of: Int32.self)!
        for (offset, element) in centroidIndices.enumerated()  {
            if element == 1 {
                indicesToPop.insert(offset)
            }
            node.entries[offset].pDistance = distances[Int(element)*entriesCount + offset]
        }
        
        var cluster2: [MTreeEntry] = []
        for index in indicesToPop.sorted(by: >) {
            let element = node.entries.remove(at: index)
            cluster2.append(element)
        }
        
        let newNode2 = newNode(entries: &cluster2, isLeaf: node.isLeaf)
        let parentEntry2 = newPEntry(mObject: centroids[1], subTreeNode: newNode2)
        
        return (centroids[0], parentEntry2)
    }
    
    private func storeEntryinNode(mainNode: MTreeNode, newEntry: MTreeEntry) {
        mainNode.entries.append(newEntry)
        newEntry.mainNode = mainNode
    }
    
    private func newNode(entries: inout [MTreeEntry], isLeaf: Bool) -> MTreeNode {
        let newNode = MTreeNode(isLeaf: isLeaf)
        newNode.entries = entries
        entries.forEach { $0.mainNode = newNode }
        return newNode
    }
    
    private func newPEntry(mObject: [Float], subTreeNode: MTreeNode) -> MTreeEntry {
        let maxParentDistance = subTreeNode.isLeaf ? subTreeNode.entries.map { $0.pDistance }.max() : subTreeNode.entries.map { $0.pDistance + $0.coverRadius}.max()
        let parentEntry = MTreeEntry(mObject: mObject, oid: UUID().uuidString, subTreeNode: subTreeNode, coverRadius: maxParentDistance!)
        subTreeNode.pEntry = parentEntry
        return parentEntry
    }
}






