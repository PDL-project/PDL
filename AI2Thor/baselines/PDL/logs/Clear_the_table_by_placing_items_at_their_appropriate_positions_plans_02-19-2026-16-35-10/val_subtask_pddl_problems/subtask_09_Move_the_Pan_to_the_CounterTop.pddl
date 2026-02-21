(define (problem move-pan-to-countertop)
  (:domain allactionrobot)

  (:objects
    robot1 - robot
    pan - object
    diningtable - object
    countertop - object
    kitchen - object
  )

  (:init
    (= (total-cost) 0)
    (not (inaction robot1))
    (at robot1 kitchen)
    (at-location pan diningtable)
    (not (holding robot1 pan))
  )

  (:goal (and
    (at-location pan countertop)
    (not (holding robot1 pan))
  ))

  (:metric minimize (total-cost))
)