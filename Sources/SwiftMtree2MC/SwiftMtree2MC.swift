// This implementation follows
// http://www-db.disi.unibo.it/research/papers/SEBD97.pdf

public struct SwiftMtree2MC {
    private var mTree: MTree
    
    public init(maxEntries: Int = 10) {
        mTree = MTree.init(maxEntries: maxEntries)
    }
    
    public func insert(mObject: [Float], oid: String) {
        mTree.insert(mObject: mObject, oid: oid)
    }
    
    public func rangeSearch(queryQbj: [Float], range: Float) -> [(String, Float)] {
        return mTree.rangeSearch(queryQbj: queryQbj, range: range).sorted(by: {$0.1 < $1.1})
    }
    
    public func knnSearch(queryQbj: [Float], k: Int) -> [(oid: String, distance: Float)] {
        return mTree.knnSearch(queryQbj: queryQbj, k: k)
    }
    
    public func toJSON() -> [String: Any] {
        return mTree.toJSON()
    }
}

extension MTree {
    
    func count(node: MTreeNode) -> Int {
        var allObj : Int = 0
        if node.isLeaf {
            allObj = node.entries.count
        } else {
            for entry in node.entries {
                allObj += count(node: entry.subTreeNode!)
            }
        }
        return allObj
    }
    
    func toJSON() -> [String: Any] {
        var json = [String: Any]()
        json["root"] = self.root.toJSON()
        return json
    }
    
    func rangeSearch(queryQbj: [Float], range: Float) -> [(String, Float)] {
        var result: [(String, Float)] = []
        for entry in self.root.entries {
            let dOrQ = distanceFunc(queryQbj, entry.mObject)
            if dOrQ <= range + entry.coverRadius {
                if let subTreeNode = entry.subTreeNode {
                    result.append(contentsOf: rangeSearch(node: subTreeNode, queryQbj: queryQbj, range: range))
                } else {
                    if dOrQ <= range {
                        result.append( (entry.oid, dOrQ) )
                    }
                }
            }
        }
        return result
    }
    
    private func rangeSearch(node: MTreeNode, queryQbj: [Float], range: Float) -> [(String, Float)] {
        var result: [(String, Float)] = []
        
        if let parentEntry = node.pEntry {
            let dOpQ = distanceFunc(queryQbj, parentEntry.mObject)
            
            if node.isLeaf {
                for entry in node.entries {
                    if abs(dOpQ - entry.pDistance) <= range {
                        let dOQ = distanceFunc(queryQbj, entry.mObject)
                        if dOQ <= range {
                            result.append((entry.oid, dOQ))
                        }
                    }
                }
            } else {
                for entry in node.entries {
                    if abs(dOpQ - entry.pDistance) <= range + entry.coverRadius {
                        let dOrQ = distanceFunc(queryQbj, entry.mObject)
                        if dOrQ <= range + entry.coverRadius {
                            result.append(contentsOf: rangeSearch(node: entry.subTreeNode!, queryQbj: queryQbj, range: range))
                        }
                    }
                }
            }
        }
        
        return result
    }
    
    func knnSearch(queryQbj: [Float], k: Int) -> [(oid: String, distance: Float)] {
        self.dk = 2147483647.0
        
        nearestNeighbors.removeAll()
        //print("count(node: self.root)", count(node: self.root))
        nearestNeighbors = Array(repeating: ("", self.dk), count: k)
        
        priorityQueue.removeAll()
        if self.root.isLeaf {
            let results: [(oid: String, distance: Float)]  = self.root.entries.map( { ( oid: $0.oid, distance: distanceFunc(queryQbj, $0.mObject) ) } ).sorted(by: { $0.distance < $1.distance })
            return Array(results[..<k])
        }
        for entry in self.root.entries {
            let dmin = max(distanceFunc(queryQbj, entry.mObject) - entry.coverRadius, 0)
            priorityQueue.append( (entry.subTreeNode!, dmin) )
        }
       
        while !priorityQueue.isEmpty {
            if let nextNode = chooseNode(&priorityQueue) {
                knnNodeSearch(node: nextNode, queryQbj: queryQbj, k: k)
            }
        }
        
        return nearestNeighbors
    }
    
    private func knnNodeSearch(node: MTreeNode, queryQbj: [Float], k: Int) {
        if let parentEntry = node.pEntry {
            let dOpQ = distanceFunc(queryQbj, parentEntry.mObject)
            if node.isLeaf {
                for entry in node.entries {
                    if abs(dOpQ - entry.pDistance) <= self.dk {
                        let dOjQ = distanceFunc(queryQbj, entry.mObject)
                        if dOjQ <= self.dk {
                            updateNN(entry.oid, dOjQ)
                            priorityQueue.removeAll { $0.dmin > self.dk }
                        }
                    }
                }
            } else {
                for entry in node.entries {
                    if abs(dOpQ - entry.pDistance) <= self.dk + entry.coverRadius {
                        let dOrQ = distanceFunc(queryQbj, entry.mObject)
                        let dmin = max(dOrQ - entry.coverRadius, 0)
                        let dmax = dOrQ + entry.coverRadius
                        
                        if dmin <= self.dk {
                            priorityQueue.append((entry.subTreeNode!, dmin))
                            if dmax < self.dk {
                                //updateNN(entry.oid, dmax)
                                //self.dk = dmax
                                priorityQueue.removeAll { $0.dmin > self.dk }
                            }
                        }
                    }
                }
            }
        }
    }
    
    private func updateNN(_ oid: String, _ distance: Float) {
        if distance < self.dk {
            nearestNeighbors.append((oid, distance))
            nearestNeighbors.sort { $0.distance < $1.distance }
            nearestNeighbors.removeLast()
            self.dk = nearestNeighbors.last!.distance
        }
    }
    
    private func chooseNode(_ priorityQueue: inout [(node: MTreeNode, dmin: Float)]) -> MTreeNode? {
        guard let (minIndex, minElement) = priorityQueue.enumerated().min(by: { $0.element.dmin < $1.element.dmin }) else {
            return nil
        }
        priorityQueue.remove(at: minIndex)
        return minElement.node
    }
}
