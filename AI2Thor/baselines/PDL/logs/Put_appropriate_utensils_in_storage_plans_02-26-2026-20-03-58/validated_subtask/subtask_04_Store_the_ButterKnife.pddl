(define (problem store-butterknife)
  (:domain allactionrobot)

  (:objects
    robot1 - robot
    butterknife - object
    drawer1 - object
    countertop - object
    floor - object
  )

  (:init
    (= (total-cost) 0)
    (not (inaction robot1))
    (at robot1 countertop)
    (at-location butterknife countertop)
    (at-location drawer1 floor)
    (object-close robot1 drawer1)
    (not (holding robot1 butterknife))
  )

  (:goal (and
    (at-location butterknife drawer1)
    (object-close robot1 drawer1)
  ))

  (:metric minimize (total-cost))
)