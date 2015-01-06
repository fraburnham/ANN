(ns nn.core)

(defn sigmoid [x]
  (/ 1 (- 1 (Math/exp (- x)))))

(defn sum [f & rest]
  (reduce + (apply (partial map f) rest)))

(defn sq-diff [a b]
  (Math/pow (- a b) 2))

(defn euclidean-distance [a b]
  (Math/sqrt (sum sq-diff a b)))

(defn sq-euclidean-distance [a b]
  (sum sq-diff a b))

;a precptron model neuron takes inputs and weights and returns
;the activation value
(defn neuron [inputs weights]
  (sigmoid (sum #(* %1 %2) inputs weights)))

;do unregularized first
;outputs
;((output 1) (output 2) (output 3))
;so that multiclass outputs are possible
(defn cost [training-outputs hypo-outputs]
  (* (/ 1 (count training-outputs))
     (sum #(sq-euclidean-distance %1 %2)
          training-outputs hypo-outputs)))

;forward prop is done in hypothesis
;network is weights matrix represented as a list
;(((neuron 1 in layer 1 weights) (neuron 2 in layer 1 weights))
; ((neuron 1 in layer 2 weights) (neuron 2 in layer 2 weights))
; ((neuron 1 in layer 3 weights) (neuron 2 in layer 3 weights)))
;so count weights is num-layers
;count weights[num-layers] is neurons-this-layer
;the inputs ARE the first layer
(defn hypothesis [inputs network]
  (loop [n network
         i inputs]
    (if (empty? n)
      (rest i) ;drop the bias
      (recur (rest n) (cons 1 (map #(neuron i %) (first n)))))))
