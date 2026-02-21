(define (problem move-pan-to-stove)
  (:domain allactionrobot)

  (:objects
    robot1 - robot
    pan - object
    diningtable - object
    stoveburner - object
    kitchen - object
  )

  (:init
    (= (total-cost) 0)
    (not (inaction robot1))
    (at robot1 kitchen)
    (at-location pan diningtable)
    (not (holding robot1 pan))
    (not (object-close robot1 stoveburner))
  )

  (:goal (and
    (at-location pan stoveburner)
    (object-close robot1 stoveburner)
  ))

  (:metric minimize (total-cost))
)