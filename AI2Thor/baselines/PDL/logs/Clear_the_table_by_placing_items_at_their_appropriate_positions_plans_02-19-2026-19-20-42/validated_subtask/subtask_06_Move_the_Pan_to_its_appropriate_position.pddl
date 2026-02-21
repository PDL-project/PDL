(define (problem move-pan-to-stove)
  (:domain allactionrobot)

  (:objects
    robot1 - robot
    pan - object
    diningTable - object
    stoveBurner - object
    kitchen - object
  )

  (:init
    (= (total-cost) 0)
    (not (inaction robot1))
    (at robot1 kitchen)
    (at-location pan diningTable)
    (not (holding robot1 pan))
    (not (object-close robot1 stoveBurner))
  )

  (:goal (and
    (at-location pan stoveBurner)
    (not (holding robot1 pan))
    (object-close robot1 stoveBurner)
  ))

  (:metric minimize (total-cost))
)