import Accelerate

extension MTree {
    
    public func testKmeans(test: [[Float]]) {
        self.dataset.removeAll()
        for entry in test {
            self.dataset.append(entry)
        }
        
        initializeCentroids()
        update()
        let oldCentroidIndices = self.centroidIndicesDescriptor.makeArray(of: Int32.self)!
        print(oldCentroidIndices)
    }
    
    // k-means
    func initializeCentroids() {
        self.centroids.removeAll()
        let dataSize = self.dataset.count
        let randomIndex = Int.random(in: 0 ..< dataSize)
        self.centroids.append(self.dataset[randomIndex])
        
        let tmp = UnsafeMutableBufferPointer(start: distances.baseAddress!,
                                             count: dataSize)
        for i in 1 ..< self.kmeans {
            for (index, data) in self.dataset.enumerated() {
                tmp[index] = distanceFunc(data, centroids[i - 1])
            }
            
            let weightedRandomIndex = weightedRandomIndex(tmp)
            self.centroids.append(self.dataset[weightedRandomIndex])
        }
    }
    
    func update() {
        var converged = false
        var iterationCount = 0
            
        while !converged && iterationCount < self.maximumIterations {
            converged = updateCentroids()
            iterationCount += 1
        }
    }
    
    private func updateCentroids() -> Bool {
        let oldCentroidIndices = self.centroidIndicesDescriptor.makeArray(of: Int32.self)!
        var oldClusterCounts = [Int]()
        for centroid in self.centroids.enumerated() {
            let indices = oldCentroidIndices.enumerated().filter {
                $0.element == centroid.offset
            }.map {
                Int($0.offset)
            }
            oldClusterCounts.append(indices.count)
        }
        
        populateDistances()
        let centroidIndices = makeCentroidIndices()
        
        var newClusterCounts = [Int]()
        for centroid in self.centroids.enumerated() {
            let indices = centroidIndices.enumerated().filter {
                $0.element == centroid.offset
            }.map {
                Int($0.offset)
            }
            
            newClusterCounts.append(indices.count)
            
            let zero: Float = 0.0
            let dimension = centroid.element.count
            var gather = Array(repeating: zero, count: dimension)

            for index in indices {
                gather = vDSP.add(gather, self.dataset[index])
            }
            
            let indicesCount = Float(indices.count)
            centroids[centroid.offset] = vDSP.divide(gather, indicesCount)
        }
        
        return oldClusterCounts.elementsEqual(newClusterCounts) { a, b in
            return abs(a - b) < self.tolerance
        }
    }
    
    private func populateDistances() {
        let dataSize = self.dataset.count
        for (k, centroid) in centroids.enumerated() {
            for (index, data) in self.dataset.enumerated() {
                distances[dataSize*k + index] = distanceFunc(data, centroid)
            }
        }
    }

    private func makeCentroidIndices() -> [Int32] {
        let dataSize = self.dataset.count
        let distancesDescriptor = BNNSNDArrayDescriptor(
            data: self.distances,
            shape: .matrixRowMajor(dataSize, self.kmeans))!
        
        let reductionLayer = BNNS.ReductionLayer(function: .argMin,
                                                 input: distancesDescriptor,
                                                 output: self.centroidIndicesDescriptor,
                                                 weights: nil)
        
        try! reductionLayer?.apply(batchSize: 1,
                                   input: distancesDescriptor,
                                   output: self.centroidIndicesDescriptor)
        
        return self.centroidIndicesDescriptor.makeArray(of: Int32.self)!
    }
    
    private func weightedRandomIndex(_ weights: UnsafeMutableBufferPointer<Float>) -> Int {
        var outputDescriptor = BNNSNDArrayDescriptor.allocateUninitialized(
            scalarType: Float.self,
            shape: .vector(1))
  
        var probabilities = BNNSNDArrayDescriptor(
            data: weights,
            shape: .vector(weights.count))!
        
        let randomGenerator = BNNSCreateRandomGenerator(
            BNNSRandomGeneratorMethodAES_CTR,
            nil)
        
        BNNSRandomFillCategoricalFloat(
            randomGenerator, &outputDescriptor, &probabilities, false)
        
        defer {
            BNNSDestroyRandomGenerator(randomGenerator)
            outputDescriptor.deallocate()
        }
        
        return Int(outputDescriptor.makeArray(of: Float.self)!.first!)
    }
}
