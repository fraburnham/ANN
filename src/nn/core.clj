(ns nn.core)

(defn sigmoid [x]
  (/ 1 (+ 1 (Math/exp (- x)))))

(defn sum [f & rest]
  (reduce + (apply (partial map f) rest)))

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
     (sum (fn [to ho] ;sum the classification errors for each training example
            (sum (fn [y hy] ;sum the error for the individual classifications
                   (+ (* y (Math/log hy))
                      (* (- 1 y) (Math/log (- 1 hy)))))
                 to ho))
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
