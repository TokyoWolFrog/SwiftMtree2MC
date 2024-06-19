import XCTest
@testable import SwiftMtree2MC

final class SwiftMtree2MCTests: XCTestCase {
    func testMTree() {
        let dimensions = 3
        let numberOfPoints = 1000

        // Generate random dataset
        let data: [[Float]] = (0..<numberOfPoints).map { _ in
            (0..<dimensions).map { _ in Float.random(in: 0.0...1.0) * 100 }
        }

        let swiftMTree = SwiftMtree2MC()
        for (index, data) in data.enumerated() {
            swiftMTree.insert(mObject: data, oid: String(index))
        }
        
        print(swiftMTree.rangeSearch(queryQbj: data[numberOfPoints/2], range: 30))        
        print(swiftMTree.knnSearch(queryQbj: data[numberOfPoints/2], k: 5))
    }
}
