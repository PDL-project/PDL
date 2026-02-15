(define (problem place-pan-on-stoveburner)
  (:domain allactionrobot)

  (:objects
    robot1 - robot
    pan - object
    stoveburner - object
    diningtable - object
  )

  (:init
    (= (total-cost) 0)
    (not (inaction robot1))
    (at robot1 diningtable)
    (at-location pan diningtable)
    (not (holding robot1 pan))
  )

  (:goal (and
    (at-location pan stoveburner)
    (not (holding robot1 pan))
  ))

  (:metric minimize (total-cost))
)