(ns nn.pso-nn
  (require [pso.core :as p]
           [pso.simple-pso :as sp]
           [nn.core :as n]))

(defn get-position [particle]
  (last particle))

(defn get-dimension [structure]
  (loop [s structure
         d 0]
    (if (<= (count s) 1)
      d
      (recur (rest s) (+ d (* (first s) (second s)))))))

(defn build-space [dimension]
  (repeatedly dimension (fn [] [-1 1])))

;structure will be simply (4 2 3 1) num of nodes in layer
;YOU MUST account for the bias neurons!!!!!
(defn position-to-nn [position structure]
  (loop [position position
         structure structure
         ret []]
    (if (empty? position)
      ret
      (let [total-weights (* (first structure) (second structure))]
        (recur (drop total-weights position)
               (rest structure)
               (conj ret (partition (first structure)
                                    (take total-weights position))))))))

(defn hypo-grains [network t-ins]
  (map #(n/hypothesis % network) t-ins))

(defn fitness [training-ins training-outs structure position]
  (let [network (position-to-nn position structure)
        hypo-outs (partition 1
                             (flatten
                               (pmap (partial hypo-grains network)
                                     (partition-all 5 training-ins))))]
    (n/cost training-outs hypo-outs)))

(defn gen-swarm [structure particle-count fitness-fn]
  (let [dimension (get-dimension structure)
        space (build-space dimension)]
    (p/generate-swarm space particle-count fitness-fn)))

(defn pso-nn [training-in training-out structure speed
              particle-count fitness-goal max-iter & {:keys [chart?]}]
  (let [swarm (gen-swarm structure particle-count
                         (partial fitness training-in training-out structure))]
    (sp/pso (build-space (get-dimension structure)) swarm speed fitness-goal
           (partial fitness training-in training-out structure) max-iter
           :chart? chart?)))
